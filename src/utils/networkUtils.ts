/**
 * Network utility functions for validating local network connections
 */

// Local network IP ranges
export const LOCAL_NETWORK_RANGES = [
  // Private IP address ranges (RFC 1918)
  { start: '10.0.0.0', end: '10.255.255.255', mask: '255.0.0.0' },
  { start: '172.16.0.0', end: '172.31.255.255', mask: '255.240.0.0' },
  { start: '192.168.0.0', end: '192.168.255.255', mask: '255.255.0.0' },
  
  // Link-local addresses
  { start: '169.254.0.0', end: '169.254.255.255', mask: '255.255.0.0' },
  
  // Loopback addresses
  { start: '127.0.0.0', end: '127.255.255.255', mask: '255.0.0.0' },
  
  // Common mobile hotspot ranges
  { start: '192.168.42.0', end: '192.168.42.255', mask: '255.255.255.0' },
  { start: '192.168.43.0', end: '192.168.43.255', mask: '255.255.255.0' },
  
  // USB tethering common ranges
  { start: '192.168.137.0', end: '192.168.137.255', mask: '255.255.255.0' },
];

/**
 * Convert IP address to 32-bit number for comparison
 */
export function ipToNumber(ip: string): number {
  const parts = ip.split('.');
  if (parts.length !== 4) {
    throw new Error(`Invalid IP address format: ${ip}`);
  }
  
  return (
    (parseInt(parts[0]) << 24) |
    (parseInt(parts[1]) << 16) |
    (parseInt(parts[2]) << 8) |
    parseInt(parts[3])
  );
}

/**
 * Convert 32-bit number back to IP address
 */
export function numberToIp(num: number): string {
  return [
    (num >>> 24) & 255,
    (num >>> 16) & 255,
    (num >>> 8) & 255,
    num & 255
  ].join('.');
}

/**
 * Check if an IP address is within a given range
 */
export function isIpInRange(ip: string, rangeStart: string, rangeEnd: string): boolean {
  try {
    const ipNum = ipToNumber(ip);
    const startNum = ipToNumber(rangeStart);
    const endNum = ipToNumber(rangeEnd);
    
    return ipNum >= startNum && ipNum <= endNum;
  } catch (error) {
    console.error('Error checking IP range:', error);
    return false;
  }
}

/**
 * Check if an IP address is in any local network range
 */
export function isLocalNetworkIp(ip: string): boolean {
  if (!ip || ip === 'localhost') {
    return true;
  }
  
  // Check if it's localhost
  if (ip === '127.0.0.1' || ip === '::1') {
    return true;
  }
  
  // Check against all local network ranges
  for (const range of LOCAL_NETWORK_RANGES) {
    if (isIpInRange(ip, range.start, range.end)) {
      return true;
    }
  }
  
  return false;
}

/**
 * Get the local network segment of an IP address
 */
export function getNetworkSegment(ip: string, mask: string = '255.255.255.0'): string {
  try {
    const ipNum = ipToNumber(ip);
    const maskNum = ipToNumber(mask);
    const networkNum = ipNum & maskNum;
    return numberToIp(networkNum);
  } catch (error) {
    console.error('Error getting network segment:', error);
    return ip;
  }
}

/**
 * Check if two IP addresses are on the same local network
 */
export function areIpsOnSameNetwork(ip1: string, ip2: string, mask: string = '255.255.255.0'): boolean {
  try {
    const segment1 = getNetworkSegment(ip1, mask);
    const segment2 = getNetworkSegment(ip2, mask);
    return segment1 === segment2;
  } catch (error) {
    console.error('Error checking if IPs are on same network:', error);
    return false;
  }
}

/**
 * Get the current device's local IP address
 */
export async function getDeviceIpAddress(): Promise<string> {
  try {
    // Try WebRTC method first
    const webrtcIp = await getLocalIpViaWebRTC();
    if (webrtcIp && webrtcIp !== '0.0.0.0' && webrtcIp !== '127.0.0.1') {
      // Sanitize and trim the IP address
      const sanitizedIp = sanitizeIpAddress(webrtcIp.trim());
      if (sanitizedIp) {
        console.log('🌐 WebRTC detected local IP:', sanitizedIp);
        return sanitizedIp;
      }
    }
    
    console.warn('🌐 WebRTC IP detection failed, using fallback');
    
    // Try to extract IP from current hostname if it's an IP address
    const hostname = window.location.hostname;
    if (hostname !== 'localhost' && !hostname.includes('.trycloudflare.com') && !hostname.includes('.cfargotunnel.com')) {
      const ipFromHostname = sanitizeIpAddress(hostname);
      if (ipFromHostname && isLocalNetworkIp(ipFromHostname)) {
        console.log('🌐 Using IP from hostname:', ipFromHostname);
        return ipFromHostname;
      }
    }
    
    // Try to get network segment from current URL and generate a likely IP
    if (hostname !== 'localhost') {
      const likelyIp = generateLikelyLocalIp(hostname);
      if (likelyIp) {
        console.log('🌐 Using generated likely IP:', likelyIp);
        return likelyIp;
      }
    }
    
    // Fallback to common local network IPs for development
    // Prioritize the IP used by ScreenShareService for consistency
    const commonLocalIps = ['192.168.254.100', '192.168.1.100', '192.168.0.100', '10.0.0.100', '172.16.0.100'];
    const fallbackIp = commonLocalIps[0];
    console.log('🌐 Using fallback IP:', fallbackIp);
    return fallbackIp;
  } catch (error) {
    console.error('Error getting device IP:', error);
    // Fallback to localhost
    return '127.0.0.1';
  }
}

/**
 * Generate a likely local IP address based on hostname
 */
function generateLikelyLocalIp(hostname: string): string | null {
  try {
    // If hostname contains numbers, try to extract network segment
    const numberMatch = hostname.match(/\d+/);
    if (numberMatch) {
      const num = parseInt(numberMatch[0]);
      if (num >= 1 && num <= 254) {
        // Try common network segments with this number
        // Prioritize the network segment used by ScreenShareService
        const segments = ['192.168.254', '192.168', '10.0', '172.16'];
        for (const segment of segments) {
          const likelyIp = `${segment}.${num}.100`;
          if (isLocalNetworkIp(likelyIp)) {
            return likelyIp;
          }
        }
      }
    }
    
    // Try to guess based on common patterns
    // Prioritize the network segment used by ScreenShareService
    if (hostname.includes('254') || hostname.includes('108')) {
      return '192.168.254.100';
    }
    if (hostname.includes('192') || hostname.includes('168')) {
      return '192.168.1.100';
    }
    if (hostname.includes('10') || hostname.includes('0')) {
      return '10.0.0.100';
    }
    if (hostname.includes('172') || hostname.includes('16')) {
      return '172.16.0.100';
    }
    
    return null;
  } catch (error) {
    console.warn('Error generating likely IP:', error);
    return null;
  }
}

/**
 * Get local IP address using WebRTC
 */
async function getLocalIpViaWebRTC(): Promise<string | null> {
  try {
    const pc = new RTCPeerConnection({
      iceServers: []
    });
    
    pc.createDataChannel('');
    
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    
    return new Promise((resolve) => {
      setTimeout(() => {
        try {
          const sdp = pc.localDescription?.sdp;
          if (sdp) {
            const lines = sdp.split('\n');
            // Look for candidate lines which contain the actual IP addresses
            for (const line of lines) {
              if (line.startsWith('a=candidate:') && line.includes('host')) {
                const parts = line.split(' ');
                if (parts.length >= 6) {
                  const ip = parts[4];
                  const sanitizedIp = sanitizeIpAddress(ip.trim());
                  if (sanitizedIp && sanitizedIp !== '0.0.0.0' && sanitizedIp !== '127.0.0.1') {
                    pc.close();
                    return resolve(sanitizedIp);
                  }
                }
              }
            }
            
            // Also check c=IN IP4 lines as fallback
            for (const line of lines) {
              if (line.startsWith('c=IN IP4 ')) {
                const ip = line.substring(9);
                const sanitizedIp = sanitizeIpAddress(ip.trim());
                if (sanitizedIp && sanitizedIp !== '0.0.0.0' && sanitizedIp !== '127.0.0.1') {
                  pc.close();
                  return resolve(sanitizedIp);
                }
              }
            }
          }
        } catch (e) {
          console.warn('WebRTC IP detection failed:', e);
        }
        
        pc.close();
        resolve(null);
      }, 200); // Increased timeout for better candidate gathering
    });
  } catch (error) {
    console.warn('WebRTC not available:', error);
    return null;
  }
}

/**
 * Get local IP addresses available on the device
 */
export async function getLocalIpAddresses(): Promise<string[]> {
  try {
    // This is a simplified version - in a real implementation, 
    // you might use WebRTC to get local network interfaces
    const localIps: string[] = [];
    
    // Add common local network IPs for development
    localIps.push('192.168.1.1', '192.168.0.1', '10.0.0.1', '172.16.0.1');
    
    return localIps;
  } catch (error) {
    console.error('Error getting local IP addresses:', error);
    return [];
  }
}

/**
 * Validate that a connection is allowed (same local network)
 */
export function validateLocalConnection(clientIp: string, serverIp: string): boolean {
  // Allow localhost connections (same machine)
  if (serverIp === 'localhost' || serverIp === '127.0.0.1') {
    return true; // Localhost is always allowed
  }
  
  // If either IP is not local, reject
  if (!isLocalNetworkIp(clientIp) || !isLocalNetworkIp(serverIp)) {
    return false;
  }
  
  // Check if they're on the same network segment
  return areIpsOnSameNetwork(clientIp, serverIp);
}

/**
 * Get network information for debugging
 */
export function getNetworkInfo(ip: string): {
  ip: string;
  isLocal: boolean;
  networkSegment: string;
  ranges: Array<{ name: string; inRange: boolean }>;
} {
  // Handle localhost specially
  if (ip === 'localhost' || ip === '127.0.0.1') {
    return {
      ip,
      isLocal: true,
      networkSegment: 'localhost',
      ranges: LOCAL_NETWORK_RANGES.map((range, index) => ({
        name: `Range ${index + 1} (${range.start} - ${range.end})`,
        inRange: false // localhost is not in any network range
      }))
    };
  }
  
  const ranges = LOCAL_NETWORK_RANGES.map((range, index) => ({
    name: `Range ${index + 1} (${range.start} - ${range.end})`,
    inRange: isIpInRange(ip, range.start, range.end)
  }));
  
  return {
    ip,
    isLocal: isLocalNetworkIp(ip),
    networkSegment: getNetworkSegment(ip),
    ranges
  };
}

/**
 * Sanitize and validate IP address
 */
export function sanitizeIpAddress(ip: string): string | null {
  if (!ip) return null;
  
  // Remove port if present
  const cleanIp = ip.split(':')[0];
  
  // Basic IP validation
  const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
  if (!ipRegex.test(cleanIp)) {
    return null;
  }
  
  // Check if each octet is valid
  const parts = cleanIp.split('.');
  for (const part of parts) {
    const num = parseInt(part);
    if (num < 0 || num > 255) {
      return null;
    }
  }
  
  return cleanIp;
}
