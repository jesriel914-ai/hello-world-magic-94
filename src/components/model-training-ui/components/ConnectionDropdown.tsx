import React, { useState, useEffect } from 'react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff, Users, UserCheck, AlertCircle, Link } from 'lucide-react';
import { DeviceInfo } from '@/services/ScreenShareService';
import { ScreenShareService } from '@/services/ScreenShareService';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import { cleanIpAddress } from '@/lib/utils';

interface ConnectionDropdownProps {
  isMobile?: boolean;
  onConnectionChange?: (isConnected: boolean, deviceInfo?: DeviceInfo) => void;
  onConnectionStatusChange?: (isConnected: boolean) => void;
}

export const ConnectionDropdown: React.FC<ConnectionDropdownProps> = ({
  isMobile = false,
  onConnectionChange,
  onConnectionStatusChange,
}) => {
  const [availableDevices, setAvailableDevices] = useState<DeviceInfo[]>([]);
  const [connectedDevice, setConnectedDevice] = useState<DeviceInfo | null>(null);
  const [showConnectionDialog, setShowConnectionDialog] = useState(false);
  const [showDisconnectDialog, setShowDisconnectDialog] = useState(false);
  const [pendingConnectionRequest, setPendingConnectionRequest] = useState<DeviceInfo | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false); // WebSocket connection status
  
  const screenShareService = ScreenShareService.getInstance();
  const { toast } = useToast();

  // Subscribe to ScreenShareService events
  useEffect(() => {
    // Subscribe to WebSocket connection status (same as mobile preview)
    screenShareService.onConnectionStatusHandler((connected) => {
      console.log('📋 Dropdown WebSocket connection status:', connected);
      setIsConnected(connected);
      if (onConnectionStatusChange) {
        onConnectionStatusChange(connected);
      }
    });
    
    // Subscribe to device list updates
    screenShareService.onDeviceListHandler((deviceListData) => {
      console.log('📋 Device list updated:', deviceListData?.devices);
      const devices = deviceListData?.devices || [];
      setAvailableDevices(devices);
      
      // Update connected device status
      const currentConnected = devices.find(device => device.isConnected);
      setConnectedDevice(currentConnected || null);
    });

    // Subscribe to connection requests
    screenShareService.onConnectionRequestHandler((request) => {
      console.log('🔗 Connection request received:', request);
      const requestingDevice = availableDevices.find(device => device.id === request.fromDeviceId);
      if (requestingDevice) {
        setPendingConnectionRequest(requestingDevice);
        setShowConnectionDialog(true);
      }
    });

    // Subscribe to connection responses
    screenShareService.onConnectionResponseHandler((response) => {
      console.log('✅ Connection response received:', response);
      setIsConnecting(false);
      if (response.accepted) {
        const connectedDevice = availableDevices.find(device => device.id === response.fromDeviceId);
        setConnectedDevice(connectedDevice || null);
        setIsConnected(true); // Set connection status when accepted
        toast({
          title: "Connection Established",
          description: `Connected to ${response.fromDeviceName}`,
        });
        onConnectionChange?.(true, connectedDevice);
      } else {
        toast({
          title: "Connection Rejected",
          description: `${response.fromDeviceName} rejected the connection request`,
          variant: "destructive",
        });
      }
    });

    // Subscribe to disconnection events
    screenShareService.onDisconnectedHandler((data) => {
      console.log('❌ Disconnected from device:', data.fromDeviceId);
      setConnectedDevice(null);
      setIsConnected(false); // Set connection status when disconnected
      toast({
        title: "Disconnected",
        description: "Connection has been terminated",
      });
      onConnectionChange?.(false);
    });

    return () => {
      screenShareService.onDeviceListHandler(null);
      screenShareService.onConnectionRequestHandler(null);
      screenShareService.onConnectionResponseHandler(null);
      screenShareService.onDisconnectedHandler(null);
    };
  }, [screenShareService, availableDevices, onConnectionChange, onConnectionStatusChange, toast]);

  // Handle connection request
  const handleConnectToDevice = (device: DeviceInfo) => {
    if (device.isConnected) return;
    
    setIsConnecting(true);
    screenShareService.sendConnectionRequest(device.id);
    toast({
      title: "Connection Request Sent",
      description: `Waiting for ${device.deviceName} to accept...`,
    });
  };

  // Handle connection response (accept/reject)
  const handleConnectionResponse = (accepted: boolean) => {
    if (!pendingConnectionRequest) return;
    
    screenShareService.sendConnectionResponse(pendingConnectionRequest.id, accepted);
    setShowConnectionDialog(false);
    setPendingConnectionRequest(null);
    
    if (accepted) {
      setConnectedDevice(pendingConnectionRequest);
      setIsConnected(true); // Set connection status when manually accepting connection
      onConnectionChange?.(true, pendingConnectionRequest);
    }
  };

  // Handle disconnection
  const handleDisconnect = () => {
    if (!connectedDevice) return;
    
    screenShareService.sendDisconnectRequest(connectedDevice.id);
    setShowDisconnectDialog(false);
    setConnectedDevice(null);
    setIsConnected(false); // Set connection status when manually disconnecting
    onConnectionChange?.(false);
  };

  // Filter devices based on current device type
  const filteredDevices = (availableDevices || []).filter(device => 
    device.deviceType !== (isMobile ? 'mobile' : 'desktop')
  );

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button 
            variant="ghost" 
            size={isMobile ? "default" : "sm"}
            className={`${isMobile ? 'h-10 w-10' : 'h-8 w-8'} p-0`}
            title="Connection"
          >
            <Link className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} ${connectedDevice ? 'text-green-500' : ''}`} />
            <span className="sr-only">Connection settings</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          {isMobile ? (
            // Mobile view: Show full device list and connection options
            <>
              <div className="flex items-center justify-between px-2 py-1.5 text-sm font-semibold">
                <span>Available Devices</span>
                <Badge variant={connectedDevice ? "default" : "secondary"}>
                  {filteredDevices.length}
                </Badge>
              </div>
              
              {filteredDevices.length === 0 ? (
                <div className="px-2 py-4 text-center text-sm text-muted-foreground">
                  <AlertCircle className="h-4 w-4 mx-auto mb-2" />
                  No devices found
                </div>
              ) : (
                filteredDevices.map((device) => (
                  <DropdownMenuItem
                    key={device.id}
                    onClick={() => {
                      if (connectedDevice && connectedDevice.id === device.id) {
                        // This device is connected, show disconnect dialog
                        setShowDisconnectDialog(true);
                      } else if (!connectedDevice) {
                        // No device connected, allow connection
                        handleConnectToDevice(device);
                      }
                      // If another device is connected, this device should not be clickable
                    }}
                    disabled={isConnecting || (connectedDevice && connectedDevice.id !== device.id)}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-2">
                      {connectedDevice && connectedDevice.id === device.id ? (
                        <Wifi className="h-4 w-4 text-green-500" />
                      ) : (
                        <WifiOff className="h-4 w-4 text-muted-foreground" />
                      )}
                      <div>
                        <div className="font-medium">{device.deviceName}</div>
                        <div className="text-xs text-muted-foreground">
                          {device.deviceType} • {cleanIpAddress(device.ipAddress)}
                        </div>
                      </div>
                    </div>
                    {connectedDevice && connectedDevice.id === device.id && (
                      <Badge variant="default" className="ml-2">
                        Connected
                      </Badge>
                    )}
                  </DropdownMenuItem>
                ))
              )}
            </>
          ) : (
            // Desktop view: Show only connection status
            <div className="px-2 py-4 text-center text-sm">
              <div className="flex items-center justify-center space-x-2 mb-2">
                {isConnected ? (
                  <>
                    <Wifi className="h-4 w-4 text-green-500" />
                    <span className="font-medium text-green-600">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium text-muted-foreground">Not Connected</span>
                  </>
                )}
              </div>
              <div className="text-xs text-muted-foreground">
                {isConnected ? 'Mobile device connected' : 'Waiting for mobile connection...'}
              </div>
            </div>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Connection Request Dialog */}
      <Dialog open={showConnectionDialog} onOpenChange={setShowConnectionDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Connection Request</DialogTitle>
            <DialogDescription>
              {pendingConnectionRequest?.deviceName} wants to connect to your device. 
              Do you want to accept this connection?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => handleConnectionResponse(false)}>
              Reject
            </Button>
            <Button onClick={() => handleConnectionResponse(true)}>
              Accept
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Disconnect Confirmation Dialog */}
      <Dialog open={showDisconnectDialog} onOpenChange={setShowDisconnectDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Disconnect Device</DialogTitle>
            <DialogDescription>
              Are you sure you want to disconnect from {connectedDevice?.deviceName}? 
              This will terminate the current session.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDisconnectDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDisconnect}>
              Disconnect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
