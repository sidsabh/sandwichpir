// Wikipedia PIR Client
//
// Wraps pir_client WASM with Wikipedia-specific logic:
//   - Article index (title -> row, offset)
//   - Brotli decompression (brotli-wasm)
//   - Article text rendering
//
// Console API:
//   window.useDirect = true/false   Toggle PIR vs direct mode
//   window.showDiagnostics = false  Hide latency/communication breakdown

import brotliInit, {
  decompress as brotliDecompressRaw,
  DecompressStream as BrotliDecompressStream,
  BrotliStreamResultCode,
} from '/vendor/brotli-wasm/brotli_wasm.js';

// ── State ──

// titles is the sorted (case-sensitive lex) list of article titles.
// rows/offsets are parallel arrays. We lowercase on the fly during search
// instead of keeping a duplicated lowercased index — saves ~256 MB on mobile.
let titles = [];
let rows = [];
let offsets = [];
let pirClient = null;
let brotliReady = false;
let serverInfo = null;
let useDirect = true;
let showDiagnostics = true;
let initTiming = {};  // one-time init costs

const $ = (id) => document.getElementById(id);

// ── Init ──

async function init() {
  try {
    const t0 = performance.now();

    setStatus('loading', 'Fetching server info...');
    const resp = await fetch('/api/info');
    serverInfo = await resp.json();
    console.log('Server:', serverInfo);

    // Init brotli-wasm
    setStatus('loading', 'Loading brotli decoder...');
    let t = performance.now();
    await brotliInit();
    brotliReady = true;
    initTiming.brotliInit = performance.now() - t;

    // Download and decompress article index
    // Download sorted TSV index (brotli compressed)
    setStatus('loading', 'Downloading article index...');
    t = performance.now();
    let indexResp = await fetch('/data/index.tsv.br');
    if (!indexResp.ok) {
      // Fallback to JSON
      indexResp = await fetch('/data/index.json.br');
    }
    if (!indexResp.ok) throw new Error('Failed to fetch article index');

    // Streaming path: pipe response body through native brotli + UTF-8
    // decoders, parse lines as they arrive, never materialize the full
    // ~300 MB decompressed buffer in memory. Falls back to one-shot
    // brotli-wasm if DecompressionStream('br') is unavailable.
    setStatus('loading', 'Streaming article index...');
    t = performance.now();

    titles = [];
    rows = [];
    offsets = [];

    let supportsBrStream = false;
    try {
      // Constructor throws if 'br' isn't a supported algorithm.
      new DecompressionStream('br');
      supportsBrStream = true;
    } catch (_) { /* fall through to one-shot path */ }

    if (supportsBrStream) {
      // Track raw bytes downloaded for diagnostics (pre-decompress).
      let bytesIn = 0;
      const counterStream = new TransformStream({
        transform(chunk, controller) {
          bytesIn += chunk.byteLength;
          controller.enqueue(chunk);
        },
      });

      const reader = indexResp.body
        .pipeThrough(counterStream)
        .pipeThrough(new DecompressionStream('br'))
        .pipeThrough(new TextDecoderStream())
        .getReader();

      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += value;
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.substring(0, nl);
          buf = buf.substring(nl + 1);
          if (line.length === 0) continue;
          const tab1 = line.indexOf('\t');
          const tab2 = line.indexOf('\t', tab1 + 1);
          titles.push(line.substring(0, tab1));
          rows.push(parseInt(line.substring(tab1 + 1, tab2), 10));
          offsets.push(parseInt(line.substring(tab2 + 1), 10));
        }
      }
      // Final line without a trailing newline
      if (buf.length > 0) {
        const tab1 = buf.indexOf('\t');
        const tab2 = buf.indexOf('\t', tab1 + 1);
        if (tab1 !== -1 && tab2 !== -1) {
          titles.push(buf.substring(0, tab1));
          rows.push(parseInt(buf.substring(tab1 + 1, tab2), 10));
          offsets.push(parseInt(buf.substring(tab2 + 1), 10));
        }
      }
      initTiming.indexBytes = bytesIn;
      initTiming.indexDownload = performance.now() - t;
      initTiming.indexDecompress = 0;  // folded into the stream pipeline
      initTiming.indexParse = 0;       // folded into the stream pipeline
    } else {
      // Streaming fallback via brotli-wasm's DecompressStream class. Used on
      // browsers that lack native DecompressionStream('br') (iOS Safari < 16.5).
      // Same memory profile as the native path: input chunks fed in, output
      // consumed line-by-line, full decompressed buffer never materialized.
      const OUTPUT_SIZE = 256 * 1024;
      const decompressor = new BrotliDecompressStream();
      const reader = indexResp.body.getReader();
      const textDec = new TextDecoder('utf-8');
      let buf = '';
      let bytesIn = 0;

      const consumeText = (chunkText) => {
        buf += chunkText;
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.substring(0, nl);
          buf = buf.substring(nl + 1);
          if (line.length === 0) continue;
          const tab1 = line.indexOf('\t');
          const tab2 = line.indexOf('\t', tab1 + 1);
          titles.push(line.substring(0, tab1));
          rows.push(parseInt(line.substring(tab1 + 1, tab2), 10));
          offsets.push(parseInt(line.substring(tab2 + 1), 10));
        }
      };

      try {
        outer: while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          bytesIn += value.byteLength;
          let input = value;
          // Drain this network chunk into the brotli decompressor, possibly
          // calling decompress() multiple times if the output buffer fills.
          while (input.byteLength > 0) {
            const result = decompressor.decompress(input, OUTPUT_SIZE);
            if (result.buf.byteLength > 0) {
              consumeText(textDec.decode(result.buf, { stream: true }));
            }
            const consumed = result.input_offset | 0;
            input = input.subarray(consumed);
            if (result.code === BrotliStreamResultCode.NeedsMoreInput) break;
            if (result.code === BrotliStreamResultCode.ResultSuccess) break outer;
            // NeedsMoreOutput: loop with remaining input
          }
        }
        // Flush TextDecoder for any pending multi-byte sequence remainders
        consumeText(textDec.decode());
        // Parse the trailing partial line, if any
        if (buf.length > 0) {
          const tab1 = buf.indexOf('\t');
          const tab2 = buf.indexOf('\t', tab1 + 1);
          if (tab1 !== -1 && tab2 !== -1) {
            titles.push(buf.substring(0, tab1));
            rows.push(parseInt(buf.substring(tab1 + 1, tab2), 10));
            offsets.push(parseInt(buf.substring(tab2 + 1), 10));
          }
        }
      } finally {
        try { decompressor.free(); } catch (_) {}
      }
      initTiming.indexBytes = bytesIn;
      initTiming.indexDownload = performance.now() - t;
      initTiming.indexDecompress = 0;
      initTiming.indexParse = 0;
    }

    initTiming.articleCount = titles.length;

    // Load WASM PIR client
    try {
      setStatus('loading', 'Loading PIR client (WASM)...');
      t = performance.now();
      const { default: initPir, PirClient } = await import('/pkg/pir_client.js');
      await initPir();
      pirClient = new PirClient(serverInfo.numItems, serverInfo.itemSizeBytes * 8);
      initTiming.pirInit = performance.now() - t;
      useDirect = false;
      console.log(`PIR client ready: ${pirClient.db_rows()} rows, ${pirClient.num_outputs()} outputs`);
    } catch (e) {
      console.warn('WASM PIR client not available, using direct mode:', e.message);
      initTiming.pirInit = null;
    }

    initTiming.total = performance.now() - t0;

    const mode = useDirect ? 'direct mode, no privacy' : 'private mode';
    setStatus('ready', `Ready (${mode}) — ${titles.length.toLocaleString()} articles`);
    updateInitDiagnostics();

    $('search').disabled = false;
    $('search').focus();
  } catch (e) {
    setStatus('error', `Init failed: ${e.message}`);
    console.error(e);
  }
}

function updateInitDiagnostics() {
  const el = $('init-diagnostics');
  if (!el || !showDiagnostics) { if (el) el.style.display = 'none'; return; }

  const t = initTiming;
  const streamed = (t.indexDecompress === 0 && t.indexParse === 0);
  const parts = streamed
    ? [
        `Index: ${(t.indexBytes / 1024 / 1024).toFixed(1)} MB streamed`,
        `parsed ${t.articleCount.toLocaleString()} articles in ${t.indexDownload.toFixed(0)} ms`,
      ]
    : [
        `Index: ${(t.indexBytes / 1024 / 1024).toFixed(1)} MB downloaded in ${t.indexDownload.toFixed(0)} ms`,
        `decompressed in ${t.indexDecompress.toFixed(0)} ms`,
        `parsed ${t.articleCount.toLocaleString()} articles in ${t.indexParse.toFixed(0)} ms`,
      ];
  if (t.pirInit !== null) {
    parts.push(`PIR client init: ${t.pirInit.toFixed(0)} ms`);
  }
  parts.push(`Total init: ${t.total.toFixed(0)} ms`);
  el.textContent = parts.join(' · ');
  el.style.display = 'block';
}

// ── Search ──

let debounce = null;
$('search').addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(updateSuggestions, 120);
});
$('search').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { const s = $('suggestions').firstChild; if (s) s.click(); }
  if (e.key === 'Escape') $('suggestions').style.display = 'none';
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrap')) $('suggestions').style.display = 'none';
});

function updateSuggestions() {
  const q = $('search').value.trim().toLowerCase();
  const el = $('suggestions');
  el.innerHTML = '';
  if (q.length < 2) { el.style.display = 'none'; return; }

  // Binary search over `titles` lowercasing on the fly. The TSV is sorted
  // case-sensitively; for case-insensitive prefix search, lowercasing each
  // probed title is fine — only ~log2(6.4M) ≈ 23 comparisons per keystroke.
  // (Saves ~256 MB by not keeping a duplicated titlesLower array.)
  let lo = 0, hi = titles.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (titles[mid].toLowerCase() < q) lo = mid + 1; else hi = mid;
  }

  // Linear walk forward collecting prefix matches.
  const matches = [];
  for (let i = lo; i < titles.length && matches.length < 15; i++) {
    if (titles[i].toLowerCase().startsWith(q)) matches.push(i);
    else break;
  }

  if (matches.length === 0) { el.style.display = 'none'; return; }

  for (const idx of matches) {
    const div = document.createElement('div');
    div.className = 'suggestion';
    div.textContent = titles[idx];
    div.onclick = () => fetchArticle(idx);
    el.appendChild(div);
  }
  el.style.display = 'block';
}

// ── Article Fetch ──

async function fetchArticle(articleIdx) {
  $('suggestions').style.display = 'none';
  $('search').value = titles[articleIdx];

  const row = rows[articleIdx];
  const offset = offsets[articleIdx];
  const title = titles[articleIdx];

  setStatus('query', `Fetching "${title}" privately...`);
  const t0 = performance.now();

  try {
    let rowBytes;
    const timing = {};

    if (!useDirect && pirClient) {
      // ── Private PIR path ──
      setStatus('query', 'Generating encrypted query...');
      let t = performance.now();
      const payload = pirClient.query(row);
      timing.queryGen = performance.now() - t;

      setStatus('query', `Sending encrypted query (${(payload.length / 1024).toFixed(0)} KB)...`);
      timing.uploadBytes = payload.length;
      t = performance.now();
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: payload,
      });
      if (!resp.ok) throw new Error(`Server: ${resp.status} ${await resp.text()}`);

      setStatus('query', 'Decoding response...');
      const encryptedResp = new Uint8Array(await resp.arrayBuffer());
      timing.network = performance.now() - t;
      timing.downloadBytes = encryptedResp.length;
      timing.serverMs = parseInt(resp.headers.get('X-Server-Time-Ms') || '0');
      timing.batchSize = parseInt(resp.headers.get('X-Batch-Size') || '1');
      timing.batchTimeout = resp.headers.get('X-Batch-Timeout') === 'true';

      t = performance.now();
      rowBytes = pirClient.decode(encryptedResp);
      timing.decode = performance.now() - t;
      timing.private = true;
    } else {
      // ── Direct path (dev, no privacy) ──
      let t = performance.now();
      const resp = await fetch(`/api/direct?row=${row}`);
      if (!resp.ok) throw new Error(`Server: ${resp.status}`);
      rowBytes = new Uint8Array(await resp.arrayBuffer());
      timing.network = performance.now() - t;
      timing.downloadBytes = rowBytes.length;
      timing.private = false;
    }

    // Extract + decompress article
    let t = performance.now();
    const articleBytes = extractArticle(rowBytes, offset);
    let text;
    try {
      text = decompressBrotli(articleBytes);
    } catch (e) {
      text = new TextDecoder().decode(articleBytes);
    }
    timing.decompress = performance.now() - t;
    timing.total = performance.now() - t0;
    timing.rowBytes = rowBytes.length;
    timing.compressedBytes = articleBytes.length;
    timing.plaintextBytes = new TextEncoder().encode(text).length;

    displayArticle(title, text, timing);
    const mode = timing.private ? 'private' : 'direct';
    setStatus('ready', `Fetched in ${timing.total.toFixed(0)} ms (${mode})`);
  } catch (e) {
    setStatus('error', `Failed: ${e.message}`);
    console.error(e);
  }
}

function extractArticle(rowData, offset) {
  if (offset + 4 > rowData.length) return new Uint8Array(0);
  const view = new DataView(rowData.buffer, rowData.byteOffset + offset, 4);
  const len = view.getUint32(0, true);
  const end = Math.min(offset + 4 + len, rowData.length);
  return rowData.slice(offset + 4, end);
}

function decompressBrotli(compressed) {
  return new TextDecoder().decode(brotliDecompressRaw(compressed));
}

function displayArticle(title, text, timing) {
  $('article-title').textContent = title;

  if (showDiagnostics) {
    const lines = [];
    const ptKB = (timing.plaintextBytes / 1024).toFixed(0);
    const compKB = (timing.compressedBytes / 1024).toFixed(0);
    const rowKB = (timing.rowBytes / 1024).toFixed(0);
    if (timing.private) {
      const wire = Math.max(0, timing.network - timing.serverMs);
      const batchNote = timing.batchTimeout ? 'timeout' : 'full';
      const pirRate = (timing.rowBytes / timing.downloadBytes * 100).toFixed(0);
      const effRate = (timing.plaintextBytes / timing.downloadBytes * 100).toFixed(0);
      lines.push('Retrieved privately via SandwichPIR');
      lines.push(`Query gen: ${timing.queryGen.toFixed(0)} ms | Server: ${timing.serverMs} ms (batch ${timing.batchSize}, ${batchNote}) | Wire: ${wire.toFixed(0)} ms | Decode: ${timing.decode.toFixed(0)} ms | Decompress: ${timing.decompress.toFixed(0)} ms | Total: ${timing.total.toFixed(0)} ms`);
      lines.push(`Upload: ${(timing.uploadBytes / 1024).toFixed(0)} KB | Download: ${(timing.downloadBytes / 1024).toFixed(0)} KB | Retrieved: ${rowKB} KB (PIR rate ${pirRate}%) | Article: ${ptKB} KB (effective rate ${effRate}%)`);
    } else {
      lines.push('Retrieved directly (dev mode, no privacy)');
      lines.push(`Network: ${timing.network.toFixed(0)} ms | Total: ${timing.total.toFixed(0)} ms | Download: ${(timing.downloadBytes / 1024).toFixed(0)} KB | Article: ${ptKB} KB (${compKB} KB compressed)`);
    }
    $('article-meta').innerHTML = lines.map(l => `<div>${l}</div>`).join('');
    $('article-meta').style.display = 'block';
  } else {
    $('article-meta').innerHTML = timing.private
      ? `Retrieved privately via SandwichPIR in ${timing.total.toFixed(0)} ms`
      : `Retrieved in ${timing.total.toFixed(0)} ms`;
    $('article-meta').style.display = 'block';
  }

  // Render: convert == Section == headers to HTML
  let html = escapeHtml(text);
  html = html.replace(/^====\s*(.+?)\s*====/gm, '</p><h4>$1</h4><p>');
  html = html.replace(/^===\s*(.+?)\s*===/gm, '</p><h3>$1</h3><p>');
  html = html.replace(/^==\s*(.+?)\s*==/gm, '</p><h2>$1</h2><p>');
  html = html.replace(/\n\n+/g, '</p><p>');
  $('article-body').innerHTML = '<p>' + html + '</p>';
  $('article').style.display = 'block';
  $('article').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Status ──

function setStatus(type, msg) {
  const el = $('status');
  el.className = 'status-' + type;
  el.textContent = msg;
}

// ── Console API ──

Object.defineProperty(window, 'useDirect', {
  get: () => useDirect,
  set: (v) => { useDirect = v; console.log(`PIR mode: ${v ? 'direct (no privacy)' : 'private'}`); },
});

// ── Diagnostics toggle ──

const diagToggle = $('diag-toggle');
if (diagToggle) {
  diagToggle.addEventListener('change', () => {
    showDiagnostics = diagToggle.checked;
    updateInitDiagnostics();
  });
}

// ── Start ──

init();
