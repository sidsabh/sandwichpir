//! SandwichPIR Client — composable PIR abstraction.
//!
//! Compiles to both native and WASM. Every query is stateless from the
//! server's perspective. The client caches the CRS-derived mask after the
//! first query to reduce subsequent download size.
//!
//!   1. `PirClient::new(num_items, item_size_bits)`
//!   2. `client.query(row_idx)` → request payload (keys + encrypted query)
//!   3. POST to server → encrypted response
//!   4. `client.decode(response)` → raw row bytes
//!
//! First query: server returns mask + body (full RLWE response).
//! Subsequent queries: client sends `has_mask=1`, server returns body only.
//! The mask is CRS-derived and identical for all clients/queries.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use spiral_rs::arith::*;
use spiral_rs::client::Client;
use spiral_rs::discrete_gaussian::DiscreteGaussian;
use spiral_rs::gadget::*;
use spiral_rs::number_theory::*;
use spiral_rs::params::*;
use spiral_rs::poly::*;

// ==================== Constants ====================

/// Public seed index for the first-dimension pseudorandom query matrix (CRS).
const SEED_0: u8 = 0;
/// Base seed for deterministic public-key generation (zeroed for reproducibility).
const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];
/// CRS seed for the InspiRING W key-switch mask (server-side rotation keys).
const W_SEED: [u8; 32] = [7; 32];
/// CRS seed for the InspiRING V key-switch mask (server-side conjugation key).
const V_SEED: [u8; 32] = [
    8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7,
];

// Wire format version
const FORMAT_FULL: u8 = 0;      // Server sends mask + body
const FORMAT_BODY_ONLY: u8 = 1; // Server sends body only (client has cached mask)

// ==================== Parameter construction ====================

fn get_seed(public_seed_idx: u8) -> [u8; 32] {
    let mut seed = STATIC_PUBLIC_SEED;
    seed[0] = public_seed_idx;
    seed
}

fn build_params(num_items: usize, item_size_bits: usize) -> Params {
    let modulus_width = 8;
    let db_cols = (item_size_bits as f64 / (2048.0 * modulus_width as f64)).ceil() as usize;
    let nu_1 = (num_items.next_power_of_two().trailing_zeros() as usize)
        .checked_sub(11)
        .unwrap_or(0);
    let noise_width = 0.5 * (2.0 * std::f64::consts::PI).sqrt();

    let moduli = vec![4294955009u64];
    let mut params = Params::init(
        2048, &moduli, noise_width,
        1, 256, u64::max(28, MIN_Q2_BITS),
        4, 4, 2, 3,
        true, nu_1, 1, 1, 0, 0,
    );
    params.instances = db_cols;
    params
}

// ==================== Packing key helpers ====================

fn generate_ksk_body<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    automorph: usize,
    mask: &PolyMatrixNTT<'a>,
    rng: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let t = params.t_exp_left;
    let tau_sk_reg = automorph_alloc(sk_reg, automorph);
    let sk_ntt = sk_reg.ntt();
    let minus_s_times_mask = &sk_ntt * &(-mask);
    let error_poly = PolyMatrixRaw::noise(params, 1, t, &DiscreteGaussian::init(params.noise_width), rng);
    let g_ntt = build_gadget(params, 1, t).ntt();
    let ksk = &tau_sk_reg.ntt() * &g_ntt;
    let body = &minus_s_times_mask + &error_poly.ntt();
    let result = &body + &ksk;
    result
}

fn condense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    if params.crt_count == 1 { return a.clone(); }
    let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = &mut res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for z in 0..params.poly_len {
                res_poly[z] = a_poly[z] | (a_poly[z + params.poly_len] << 32);
            }
        }
    }
    res
}

// ==================== WASM PIR Client ====================

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct PirClient {
    db_rows: usize,
    num_rlwe_outputs: usize,
    q_prime_1: u64,  // 2^10 (body modulus)
    q_prime_2: u64,  // 2^18 (mask modulus)
    cached_mask: Option<Vec<u8>>,  // cached mask rows from first query
}

static mut PARAMS_STORE: Option<Box<Params>> = None;
static mut CLIENT_STORE: Option<Client<'static>> = None;

fn static_params() -> &'static Params {
    unsafe { PARAMS_STORE.as_ref().unwrap().as_ref() }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl PirClient {
    /// Create a PIR client. Call once; reuse for multiple queries.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(num_items: usize, item_size_bits: usize) -> PirClient {
        unsafe { PARAMS_STORE = Some(Box::new(build_params(num_items, item_size_bits))); }
        let params = static_params();
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        PirClient {
            db_rows,
            num_rlwe_outputs: db_cols / params.poly_len,
            q_prime_1: 1 << 10,
            q_prime_2: 1 << 18,
            cached_mask: None,
        }
    }

    /// Whether the client has a cached mask (subsequent queries are smaller).
    pub fn has_mask(&self) -> bool {
        self.cached_mask.is_some()
    }

    /// Generate a PIR request for the given row.
    ///
    /// Wire format (little-endian):
    ///   [1B]  format flag: 0 = request full response, 1 = body only (has cached mask)
    ///   [4B]  y_body_len (u64 count)
    ///   [4B]  z_body_len (u64 count)
    ///   [y_body_len * 8B] y_body condensed
    ///   [z_body_len * 8B] z_body condensed
    ///   [db_rows * 4B] encrypted query (u32 values in Z_{2^32})
    pub fn query(&self, target_row: usize) -> Vec<u8> {
        let params = static_params();
        let poly_len = params.poly_len;
        let q = params.modulus;
        let dim_log2 = params.db_dim_1;

        // Fresh secret key per query
        let mut client = Client::init(params);
        client.generate_secret_keys();
        let sk_reg = client.get_sk_reg();

        // Fresh packing keys per query
        let gen: usize = 5;
        let gen_pow_1 = exponentiate_uint_mod(gen as u64, 1, 2 * poly_len as u64) as usize;

        let w_mask = PolyMatrixNTT::random_rng(params, 1, params.t_exp_left, &mut ChaCha20Rng::from_seed(W_SEED));
        let v_mask = PolyMatrixNTT::random_rng(params, 1, params.t_exp_left, &mut ChaCha20Rng::from_seed(V_SEED));

        let y_body = generate_ksk_body(params, &sk_reg, gen_pow_1, &w_mask, &mut ChaCha20Rng::from_entropy());
        let z_body = generate_ksk_body(params, &sk_reg, 2 * poly_len - 1, &v_mask, &mut ChaCha20Rng::from_entropy());

        let y_condensed = condense_matrix(params, &y_body);
        let z_condensed = condense_matrix(params, &z_body);

        // RLWE query
        let scale_k = q / params.pt_modulus;
        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let mut q_vals = vec![0u64; self.db_rows];

        for i in 0..(1 << dim_log2) {
            let mut scalar = PolyMatrixRaw::zero(params, 1, 1);
            if i == target_row / poly_len {
                scalar.data[target_row % poly_len] = scale_k;
            }
            let ct = client.encrypt_matrix_reg(&scalar.ntt(), &mut ChaCha20Rng::from_entropy(), &mut rng_pub);
            let ct_raw = ct.raw();
            let b_poly = ct_raw.get_poly(1, 0);
            for j in 0..poly_len {
                q_vals[i * poly_len + j] = b_poly[j];
            }
        }

        // Modswitch Q -> W=2^32
        let wq: Vec<u32> = q_vals.iter().map(|&v| {
            ((v as u128 * (1u128 << 32) + (q as u128 / 2)) / q as u128) as u32
        }).collect();

        // Build payload — all values fit in u32 (Q < 2^32, W = 2^32)
        let y_data = y_condensed.as_slice();
        let z_data = z_condensed.as_slice();
        let format_flag = if self.cached_mask.is_some() { FORMAT_BODY_ONLY } else { FORMAT_FULL };

        let mut payload = Vec::new();
        payload.push(format_flag);
        payload.extend_from_slice(&(y_data.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(z_data.len() as u32).to_le_bytes());
        for &v in y_data { payload.extend_from_slice(&(v as u32).to_le_bytes()); }
        for &v in z_data { payload.extend_from_slice(&(v as u32).to_le_bytes()); }
        for &v in &wq { payload.extend_from_slice(&v.to_le_bytes()); }

        // Store client for decode
        unsafe { CLIENT_STORE = Some(client); }

        payload
    }

    /// Decode the server's encrypted response into raw row bytes.
    /// Caches the mask from the first full response for subsequent queries.
    pub fn decode(&mut self, response_bytes: &[u8]) -> Vec<u8> {
        let params = static_params();
        let client = unsafe { CLIENT_STORE.as_ref().expect("Call query() first") };

        let q1_bits = (self.q_prime_2 as f64).log2().ceil() as usize; // 18
        let q2_bits = (self.q_prime_1 as f64).log2().ceil() as usize; // 10
        let mask_bytes_per_output = (q1_bits * params.poly_len + 7) / 8;  // row 0
        let body_bytes_per_output = (q2_bits * params.poly_len + 7) / 8;  // row 1
        let full_bytes_per_output = ((q1_bits + q2_bits) * params.poly_len + 7) / 8;

        // Determine if this is a full or body-only response
        let is_body_only = self.cached_mask.is_some()
            && response_bytes.len() < self.num_rlwe_outputs * full_bytes_per_output;

        let mut plaintext: Vec<u64> = Vec::new();

        for i in 0..self.num_rlwe_outputs {
            let ct = if is_body_only {
                // Reconstruct full RLWE from cached mask + new body
                let mask = self.cached_mask.as_ref().unwrap();
                let mask_start = i * mask_bytes_per_output;
                let body_start = i * body_bytes_per_output;
                let body_end = body_start + body_bytes_per_output;
                if body_end > response_bytes.len() { break; }

                recover_poly_split(
                    params,
                    self.q_prime_1,
                    self.q_prime_2,
                    &mask[mask_start..mask_start + mask_bytes_per_output],
                    &response_bytes[body_start..body_end],
                )
            } else {
                // Full response: mask + body interleaved
                let start = i * full_bytes_per_output;
                let end = start + full_bytes_per_output;
                if end > response_bytes.len() { break; }
                recover_poly(params, self.q_prime_1, self.q_prime_2, &response_bytes[start..end])
            };

            let dec = client.decrypt_matrix_reg(&ct.ntt()).raw();
            for z in 0..params.poly_len {
                plaintext.push(rescale(dec.data[z], params.modulus, params.pt_modulus));
            }
        }

        // Cache mask from first full response
        if !is_body_only && self.cached_mask.is_none() {
            let mut mask_cache = Vec::with_capacity(self.num_rlwe_outputs * mask_bytes_per_output);
            for i in 0..self.num_rlwe_outputs {
                let start = i * full_bytes_per_output;
                // Mask is the first q1_bits * poly_len bits of each output
                mask_cache.extend_from_slice(&response_bytes[start..start + mask_bytes_per_output]);
            }
            self.cached_mask = Some(mask_cache);
        }

        plaintext.iter().map(|&v| v as u8).collect()
    }

    pub fn db_rows(&self) -> usize { self.db_rows }
    pub fn num_outputs(&self) -> usize { self.num_rlwe_outputs }
}

// ==================== Recovery helpers ====================

/// Recover full RLWE from interleaved mask+body bytes (first query).
fn recover_poly<'a>(params: &'a Params, q_1: u64, q_2: u64, ciphertext: &[u8]) -> PolyMatrixRaw<'a> {
    let q_1_bits = (q_2 as f64).log2().ceil() as usize;
    let q_2_bits = (q_1 as f64).log2().ceil() as usize;

    let mut res = PolyMatrixRaw::zero(params, 2, 1);
    let mut bit_offs = 0;
    let (row_0, row_1) = res.data.as_mut_slice().split_at_mut(params.poly_len);

    for z in 0..params.poly_len {
        let val = read_bits(ciphertext, bit_offs, q_1_bits);
        row_0[z] = rescale(val, q_2, params.modulus);
        bit_offs += q_1_bits;
    }
    for z in 0..params.poly_len {
        let val = read_bits(ciphertext, bit_offs, q_2_bits);
        row_1[z] = rescale(val, q_1, params.modulus);
        bit_offs += q_2_bits;
    }
    res
}

/// Recover full RLWE from separate mask and body byte arrays (subsequent queries).
fn recover_poly_split<'a>(
    params: &'a Params, q_1: u64, q_2: u64,
    mask_bytes: &[u8], body_bytes: &[u8],
) -> PolyMatrixRaw<'a> {
    let q_1_bits = (q_2 as f64).log2().ceil() as usize;
    let q_2_bits = (q_1 as f64).log2().ceil() as usize;

    let mut res = PolyMatrixRaw::zero(params, 2, 1);
    let (row_0, row_1) = res.data.as_mut_slice().split_at_mut(params.poly_len);

    let mut bit_offs = 0;
    for z in 0..params.poly_len {
        let val = read_bits(mask_bytes, bit_offs, q_1_bits);
        row_0[z] = rescale(val, q_2, params.modulus);
        bit_offs += q_1_bits;
    }
    bit_offs = 0;
    for z in 0..params.poly_len {
        let val = read_bits(body_bytes, bit_offs, q_2_bits);
        row_1[z] = rescale(val, q_1, params.modulus);
        bit_offs += q_2_bits;
    }
    res
}

fn read_bits(data: &[u8], bit_offs: usize, num_bits: usize) -> u64 {
    let byte_pos = bit_offs / 8;
    let mut bit_pos = bit_offs % 8;
    let mut result: u64 = 0;
    let mut remaining = num_bits;

    for i in byte_pos..data.len() {
        if remaining == 0 { break; }
        let can_take = std::cmp::min(8 - bit_pos, remaining);
        let value = if can_take < 8 {
            (data[i] >> bit_pos) & ((1 << can_take) - 1)
        } else {
            data[i] >> bit_pos
        };
        result |= (value as u64) << (num_bits - remaining);
        remaining -= can_take;
        bit_pos = 0;
    }
    result
}
