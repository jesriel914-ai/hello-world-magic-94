#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting Signature AI Training Development Environment\n');

// Start backend server
console.log('📡 Starting backend server on port 8000...');
const backend = spawn('node', ['server.js'], {
  cwd: path.join(__dirname, 'backend'),
  stdio: 'inherit',
  shell: true
});

backend.on('error', (error) => {
  console.error('❌ Failed to start backend server:', error.message);
  console.log('💡 Make sure you have installed backend dependencies:');
  console.log('   cd backend && npm install');
});

// Start frontend server
console.log('🌐 Starting frontend server on port 5173...');
const frontend = spawn('npm', ['run', 'dev'], {
  cwd: __dirname,
  stdio: 'inherit',
  shell: true
});

frontend.on('error', (error) => {
  console.error('❌ Failed to start frontend server:', error.message);
  console.log('💡 Make sure you have installed frontend dependencies:');
  console.log('   npm install');
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down servers...');
  backend.kill('SIGINT');
  frontend.kill('SIGINT');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down servers...');
  backend.kill('SIGTERM');
  frontend.kill('SIGTERM');
  process.exit(0);
});

console.log('\n✅ Both servers are starting...');
console.log('📡 Backend API: http://localhost:8000');
console.log('🌐 Frontend: http://localhost:5173');
console.log('\nPress Ctrl+C to stop both servers');