import https from 'https';
import http from 'http';
import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import selfsigned from 'selfsigned';
import { fileURLToPath } from 'url';

// Get current directory for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Generate self-signed certificate
const attrs = [{ name: 'commonName', value: 'localhost' }];
const pems = selfsigned.generate(attrs, { days: 365 });

// Save certificate to file for download
const certPath = path.join(__dirname, 'localhost.pem');
const keyPath = path.join(__dirname, 'localhost-key.pem');

fs.writeFileSync(certPath, pems.cert);
fs.writeFileSync(keyPath, pems.private);

console.log('Certificate saved to:');
console.log(`  Cert: ${certPath}`);
console.log(`  Key: ${keyPath}`);

const options = {
  key: pems.private,
  cert: pems.cert
};

// Create HTTPS server
const server = https.createServer(options, (req, res) => {
  // Handle certificate download
  if (req.url === '/cert.pem') {
    res.writeHead(200, {
      'Content-Type': 'application/x-x509-ca-cert',
      'Content-Disposition': 'attachment; filename="localhost.crt"'
    });
    res.end(pems.cert);
    return;
  }
  
  // Handle certificate acceptance page
  if (req.url === '/cert-accept.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Install Certificate</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .step { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .download-btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; }
        .download-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>Install HTTPS Certificate</h1>
    <p>To permanently trust this certificate and stop seeing security warnings:</p>
    
    <div class="step">
        <h3>Step 1: Download Certificate</h3>
        <a href="/cert.pem" class="download-btn" download="localhost.crt">Download Certificate</a>
    </div>
    
    <div class="step">
        <h3>Step 2: Install Certificate</h3>
        <p><strong>Windows:</strong></p>
        <ol>
            <li>Double-click the downloaded localhost.crt file</li>
            <li>Click "Install Certificate"</li>
            <li>Select "Local Machine" → Next</li>
            <li>Choose "Place all certificates in the following store" → Browse</li>
            <li>Select "Trusted Root Certification Authorities" → OK</li>
            <li>Click Next → Finish</li>
        </ol>
    </div>
    
    <div class="step">
        <h3>Step 3: Restart Browser</h3>
        <p>Close and restart your browser completely.</p>
    </div>
    
    <div class="step">
        <h3>Step 4: Test</h3>
        <p>Visit <a href="https://192.168.254.100:8443">https://192.168.254.100:8443</a> - no warning should appear.</p>
    </div>
</body>
</html>
    `);
    return;
  }
  
  // Proxy the request to the Vite dev server using HTTP
  const proxy = http.request({
    hostname: 'localhost',
    port: 3000, // Vite dev server runs on port 3000
    path: req.url,
    method: req.method,
    headers: req.headers
  }, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  req.pipe(proxy, { end: true });

  // Handle proxy errors
  proxy.on('error', (err) => {
    console.error('Proxy error:', err.message);
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('Bad Gateway: Vite server not running');
  });
});

// Handle WebSocket upgrades for Vite HMR
server.on('upgrade', (req, socket, head) => {
  // Proxy WebSocket connection to Vite dev server
  const proxy = http.request({
    hostname: 'localhost',
    port: 3000,
    path: req.url,
    method: req.method,
    headers: req.headers
  });

  proxy.on('upgrade', (proxyRes, proxySocket, proxyHead) => {
    // Send the upgrade response to the client
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
      'Connection: Upgrade\r\n' +
      'Upgrade: websocket\r\n' +
      proxyRes.rawHeaders.map((header, index) => {
        return index % 2 === 0 ? `${header}: ${proxyRes.rawHeaders[index + 1]}` : '';
      }).filter(Boolean).join('\r\n') +
      '\r\n\r\n'
    );

    // Proxy the WebSocket data
    proxySocket.pipe(socket);
    socket.pipe(proxySocket);
  });

  proxy.on('error', (err) => {
    console.error('WebSocket proxy error:', err.message);
    socket.end();
  });

  proxy.end();
});

// Start the server with delay to ensure Vite is ready
const PORT = 8443;
setTimeout(() => {
  server.listen(PORT, '0.0.0.0', async () => {
    console.log('\n  VITE v7.1.3  ready in 365 ms');
    console.log('');
    console.log(`  ➜  Local:   https://localhost:${PORT}/`);
    
    const networkAddresses = await getAllNetworkAddresses();
    for (const address of networkAddresses) {
      console.log(`  ➜  Network: https://${address}:${PORT}/`);
    }
    
    console.log('');
  });
}, 3000); // Wait 3 seconds for Vite to start

// Get all network addresses
async function getAllNetworkAddresses() {
  const { networkInterfaces } = await import('os');
  const nets = networkInterfaces();
  const addresses = [];
  
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      // Skip over non-IPv4 and internal (i.e. 127.0.0.1) addresses
      if (net.family === 'IPv4' && !net.internal) {
        addresses.push(net.address);
      }
    }
  }
  
  return addresses;
}

// Start Vite dev server (frontend only)
console.log('Starting Vite dev server...');
const vite = exec('npx vite', { stdio: 'inherit' });

// Handle process termination
process.on('SIGINT', () => {
  console.log('Shutting down servers...');
  server.close();
  vite.kill();
  process.exit();
});
