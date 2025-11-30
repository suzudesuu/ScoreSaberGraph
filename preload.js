// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  chooseOutputDir: () => ipcRenderer.invoke('choose-output-dir'),
  generateGraph: (options) => ipcRenderer.invoke('generate-graph', options),
});
