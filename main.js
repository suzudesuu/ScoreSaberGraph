// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 650,
    resizable: false,
    backgroundColor: '#020617',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: 'ScoreSaber Graph Generator',
  });

  win.setMenuBarVisibility(false);
  win.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ---------- IPC: выбор папки ----------

ipcMain.handle('choose-output-dir', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });

  if (result.canceled || !result.filePaths.length) {
    return null;
  }
  return result.filePaths[0];
});

// ---------- IPC: запуск Python ----------

ipcMain.handle('generate-graph', async (event, options) => {
  return new Promise((resolve, reject) => {
    try {
      const pythonPath = 'python'; // при необходимости тут можно указать полный путь
      const scriptPath = path.join(__dirname, 'scoresaber_graph_cli.py');

      // САМЫЙ ВАЖНЫЙ МОМЕНТ: stringify объекта в JSON-строку
      const args = [scriptPath, JSON.stringify(options)];

      const child = spawn(pythonPath, args, {
        cwd: __dirname,
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8', // <— ключевая строчка
        },
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        if (code === 0) {
          resolve(stdout.trim());
        } else {
          reject(
            new Error(
              `Python exited with code ${code}\n${stderr || stdout}`.trim()
            )
          );
        }
      });

      child.on('error', (err) => {
        reject(err);
      });
    } catch (err) {
      reject(err);
    }
  });
});
