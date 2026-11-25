const indexVersion = 'e5-small-q8-v1';
let modelPipeline;

async function loadModel(model, weightsFile) {
  if (modelPipeline) return;
  self.postMessage({type: 'model', state: 'loading'});
  const {env, pipeline} = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2');
  env.allowLocalModels = false;
  // GitHub Pages does not give this app a cross-origin-isolated context.
  env.backends.onnx.wasm.numThreads = 1;
  let lastUpdate = -Infinity;
  modelPipeline = await pipeline('feature-extraction', model, {
    device: 'wasm',
    dtype: 'q8',
    progress_callback(event) {
      if (event.file !== weightsFile) return;
      if (event.status === 'download' || event.status === 'done') {
        self.postMessage({type: 'model', state: event.status === 'done' ? 'initializing' : 'downloading'});
      } else if (event.status === 'progress' && performance.now() - lastUpdate >= 150) {
        lastUpdate = performance.now();
        self.postMessage({type: 'model', state: 'downloading', loaded: event.loaded, total: event.total});
      }
    },
  });
  self.postMessage({type: 'model', state: 'ready'});
}

async function embed(text) {
  const output = await modelPipeline(text, {pooling: 'mean', normalize: true});
  return output.data;
}

let indexDatabase;
function database() {
  indexDatabase ??= new Promise((resolve, reject) => {
    const request = indexedDB.open('fzmapp', 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return indexDatabase;
}

async function getVectors(records) {
  const db = await database();
  const signature = JSON.stringify(records);
  const cached = await new Promise((resolve, reject) => {
    const request = db.transaction('data').objectStore('data').get('vectors');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  if (cached?.version === indexVersion && cached.signature === signature && cached.vectors?.length === records.length) {
    return cached.vectors;
  }

  const vectors = [];
  const byPassage = new Map();
  self.postMessage({type: 'index', count: 0, total: records.length});
  for (const [index, record] of records.entries()) {
    const passage = 'passage: ' + record.title + ' ' + record.abstract;
    if (!byPassage.has(passage)) byPassage.set(passage, await embed(passage));
    vectors.push(byPassage.get(passage));
    const count = index + 1;
    if (count % 8 === 0 || count === records.length) {
      self.postMessage({type: 'index', count, total: records.length});
    }
  }
  try {
    await new Promise((resolve, reject) => {
      const transaction = db.transaction('data', 'readwrite');
      transaction.objectStore('data').put({version: indexVersion, signature, vectors}, 'vectors');
      transaction.oncomplete = resolve;
      transaction.onerror = transaction.onabort = () => reject(transaction.error);
    });
  } catch (error) {
    // A storage limit can prevent future reuse without invalidating this search.
    console.warn('Could not cache search vectors:', error);
  }
  return vectors;
}

self.onmessage = async ({data: {model, weightsFile, queryText, records}}) => {
  try {
    await loadModel(model, weightsFile);
    const vectors = await getVectors(records);
    self.postMessage({type: 'ranking'});
    const query = await embed('query: ' + queryText);
    const groups = new Map();
    for (const [index, record] of records.entries()) {
      const score = query.reduce((sum, value, i) => sum + value * vectors[index][i], 0);
      if (!groups.has(record.faculty)) groups.set(record.faculty, []);
      groups.get(record.faculty).push({...record, score});
    }
    const results = Array.from(groups, ([name, items]) => {
      items.sort((a, b) => b.score - a.score);
      const top = items.slice(0, 5);
      const average = top.reduce((sum, item) => sum + item.score, 0) / top.length;
      return {name, items: top.slice(0, 3), score: .6 * top[0].score + .4 * average};
    }).sort((a, b) => b.score - a.score);
    self.postMessage({type: 'result', results});
  } catch (error) {
    self.postMessage({type: 'error', message: error.message, modelReady: Boolean(modelPipeline)});
  }
};
