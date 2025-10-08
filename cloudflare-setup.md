# Cloudflare Tunnel Setup Guide

## Quick Setup

### 1. Install Cloudflared
```bash
# Windows (winget)
winget install --id Cloudflare.cloudflared

# Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
```

### 2. Login to Cloudflare
```bash
cloudflared tunnel login
```
- This opens a browser window
- Login with your Cloudflare account (free is fine)
- Select your domain if you have one

### 3. Start Your Servers
```bash
# Terminal 1: Start all servers
npm run dev
```
This will start:
- Frontend server on port 5173
- Backend API on port 5173  
- WebSocket server on port 5173

**All services now run on a single port (5173) for simplified setup!**

### 4. Create Single Cloudflare Tunnel
```bash
# Terminal 2: Single tunnel for all services (port 5173)
cloudflared tunnel --url http://localhost:5173
```

**Important**: Use a single tunnel for the frontend. Cloudflare will automatically handle WebSocket connections through the same tunnel.

### 5. Access Your Application
- Cloudflare will give you a URL like: `https://random-name.trycloudflare.com`
- **Desktop**: Use `http://localhost:5173/model-training-signature-classify`
- **Mobile**: Use the Cloudflare URL: `https://random-name.trycloudflare.com/model-training-signature-classify`

The WebSocket connection will automatically work through the same tunnel.

## How It Works

✅ **No Certificate Issues**: Cloudflare provides trusted HTTPS certificates  
✅ **No Account Limits**: Free and generous usage limits  
✅ **Secure Connections**: All traffic goes through HTTPS/WSS  
✅ **Easy Setup**: No complex configuration needed  

## Mobile Testing

1. Open the Cloudflare URL on your mobile device
2. The app should load without certificate warnings
3. WebSocket connection will automatically use the secure tunnel
4. Real-time screen sharing should work immediately

## Troubleshooting

**If tunnel doesn't start:**
- Make sure your local servers are running first
- Check if ports 8443 and 8444 are available
- Try different ports if needed

**If connection fails:**
- Check that both tunnels are running
- Verify you're using the correct Cloudflare URL
- Look at the cloudflared terminal output for errors

**Alternative Setup (Single Tunnel):**
If you want to use a single tunnel for both services, you can configure a Cloudflare Tunnel with multiple origins in the dashboard.
