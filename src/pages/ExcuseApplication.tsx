import { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useUnsavedChanges } from "@/hooks/useUnsavedChanges";
import { UnsavedChangesDialog } from "@/components/UnsavedChangesDialog";
import PageWrapper from "@/components/PageWrapper";
import ExcuseForm from "@/components/ExcuseForm";

import { 
  Calendar as CalendarIcon,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Download,
  Edit,
  Eye,
  EyeOff,
  FileImage,
  List,
  MoreHorizontal,
  Plus,
  Search,
  Trash2,
  Upload,
  User,
  ChevronsUp,
  ChevronsDown,
  ChevronsUpDown,
  CheckCircle2,
  AlertCircle,
  X,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/lib/supabase";
import Layout from "@/components/Layout";
import { cn } from "@/lib/utils";

type ExcuseStatus = 'pending' | 'approved' | 'rejected';

type ExcuseApplication = {
  id: number;
  student_id: number;
  session_id?: number;
  absence_date: string;
  
  documentation_url?: string;
  status: ExcuseStatus;
  reviewed_by?: string;
  reviewed_at?: string;
  review_notes?: string;
  created_at: string;
  updated_at: string;
  // Related data
  students?: {
    id: number;
    firstname: string;
    surname: string;
    student_id: string;
    program: string;
    year: string;
  };
  sessions?: {
    id: number;
    title: string;
    date: string;
  };
};

type ExcuseFormData = {
  student_id: string;
  session_id?: string;
  absence_date: string;
  excuse_image?: File;
  documentation_url?: string;
};

const ExcuseApplicationContent = () => {
  const { toast } = useToast();
  const [excuses, setExcuses] = useState<ExcuseApplication[]>([]);
  const [loading, setLoading] = useState(true);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);
  const [selectedExcuse, setSelectedExcuse] = useState<ExcuseApplication | null>(null);
  const [editingExcuse, setEditingExcuse] = useState<ExcuseApplication | null>(null);
  const [imageZoom, setImageZoom] = useState(1);
  const [imagePan, setImagePan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [viewMode, setViewMode] = useState<'view' | 'edit'>('view');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<number | null>(null);
  const [displayPageSize, setDisplayPageSize] = useState(10);
  // Sorting
  type ExcuseSortKey = 'name' | 'id' | 'date' | 'status';
  const [sortKey, setSortKey] = useState<ExcuseSortKey>('name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [totalExcusesCount, setTotalExcusesCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');

  const {
    showConfirmDialog,
    markAsChanged,
    markAsSaved,
    handleClose,
    confirmClose,
    cancelClose,
    handleOpenChange,
  } = useUnsavedChanges({
    onClose: () => {
      setIsFormOpen(false);
      setEditingExcuse(null);
    },
    enabled: isFormOpen,
  });

  useEffect(() => {
    fetchExcuses();
  }, []);

  const handleFormSuccess = () => {
    fetchExcuses();
    setIsFormOpen(false);
    setEditingExcuse(null);
    toast({
      title: "Success",
      description: editingExcuse ? "Excuse application updated successfully" : "Excuse application submitted successfully",
    });
  };

  const fetchExcuses = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('excuse_applications')
        .select(`
          *,
          students!student_id (
            id,
            firstname,
            surname,
            student_id,
            program,
            year,
          ),
          sessions!session_id (
            id,
            title,
            date
          )
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;
      console.log('Fetched excuses data:', data);
      const excusesData = (data || []).map(excuse => ({
        ...excuse,
        status: excuse.status as ExcuseStatus
      }));
      setExcuses(excusesData);
      setTotalExcusesCount(excusesData.length);
    } catch (error) {
      console.error('Error fetching excuses:', error);
      toast({
        title: "Error",
        description: "Failed to fetch excuse applications",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle page size change
  const handlePageSizeChange = (newPageSize: number) => {
    // Allow very large numbers for "ALL" case, otherwise ensure minimum value of 10
    const validPageSize = newPageSize >= 999999 ? newPageSize : Math.max(10, newPageSize);
    
    // Update display page size (what user sees in the control)
    if (newPageSize >= 999999) {
      setDisplayPageSize(totalExcusesCount);
    } else {
      setDisplayPageSize(validPageSize);
    }
  };

  // Update display page size when total excuses count changes (for "ALL" case)
  useEffect(() => {
    if (displayPageSize >= 999999) {
      setDisplayPageSize(totalExcusesCount);
    }
  }, [totalExcusesCount, displayPageSize]);


  const handleUpdateStatus = async (id: number, status: ExcuseStatus, notes?: string) => {
    try {
      const { error } = await supabase
        .from('excuse_applications')
        .update({
          status,
          review_notes: notes,
          reviewed_at: new Date().toISOString(),
        })
        .eq('id', id);

      if (error) throw error;

      toast({
        title: "Success",
        description: `Excuse application ${status}`,
      });

      fetchExcuses();
      setIsViewOpen(false);
    } catch (error) {
      console.error('Error updating status:', error);
      toast({
        title: "Error",
        description: "Failed to update status",
        variant: "destructive",
      });
    }
  };

  const handleDeleteExcuse = async (id: number) => {
    try {
      console.log('Deleting excuse with ID:', id, 'Type:', typeof id);
      
      // First, let's check what the actual ID looks like in the database
      const { data: checkData, error: checkError } = await supabase
        .from('excuse_applications')
        .select('id')
        .limit(1);
      
      if (checkError) {
        console.error('Error checking ID format:', checkError);
      } else {
        console.log('Sample ID from database:', checkData?.[0]?.id, 'Type:', typeof checkData?.[0]?.id);
      }
      
      // Try the delete operation
      const { error, count } = await supabase
        .from('excuse_applications')
        .delete()
        .eq('id', id);

      console.log('Delete result - Error:', error, 'Count:', count);
      
      // Also try to fetch the record to see if it exists
      const { data: checkRecord, error: checkRecordError } = await supabase
        .from('excuse_applications')
        .select('*')
        .eq('id', id)
        .single();
      
      console.log('Record check - Data:', checkRecord, 'Error:', checkRecordError);

      if (error) {
        console.error('Supabase delete error:', error);
        throw error;
      }

      if (count === 0) {
        console.log('No records were deleted, trying alternative approach...');
        
        // Try alternative delete approach
        const { error: altError, count: altCount } = await supabase
          .from('excuse_applications')
          .delete()
          .eq('id', id);
        
        console.log('Alternative delete result - Error:', altError, 'Count:', altCount);
        
        if (altCount === 0) {
          toast({
            title: "Warning",
            description: "No record found to delete",
            variant: "destructive",
          });
          return;
        }
      }

      console.log('Delete successful, refreshing list...');
      
      toast({
        title: "Success",
        description: "Excuse application deleted successfully",
      });

      // Refresh the list immediately
      await fetchExcuses();
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
    } catch (error) {
      console.error('Error deleting excuse:', error);
      toast({
        title: "Error",
        description: "Failed to delete excuse application",
        variant: "destructive",
      });
    }
  };

  const handleImageMouseDown = (e: React.MouseEvent) => {
    if (imageZoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - imagePan.x, y: e.clientY - imagePan.y });
    }
  };

  const handleImageMouseMove = (e: React.MouseEvent) => {
    if (isDragging && imageZoom > 1) {
      // Add drag sensitivity control - reduce movement by dividing by zoom level
      const sensitivity = 1 / imageZoom;
      setImagePan({
        x: (e.clientX - dragStart.x) * sensitivity,
        y: (e.clientY - dragStart.y) * sensitivity
      });
    }
  };

  const handleImageMouseUp = () => {
    setIsDragging(false);
  };

  // Reset pan position when zoom changes
  const handleZoomChange = (newZoom: number) => {
    setImageZoom(newZoom);
    // Reset pan position when zoom returns to 100% or below
    if (newZoom <= 1) {
      setImagePan({ x: 0, y: 0 });
    }
  };

  const getStatusDisplay = (status: ExcuseStatus) => {
    return (
      <span className="text-xs text-gray-500 capitalize">
        {status}
      </span>
    );
  };

  // Status Badge Component (for view dialog)
  const getStatusBadge = (status: ExcuseStatus) => {
    const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
    
    switch (status) {
      case 'approved':
        return (
          <span className={`${baseClasses} bg-green-100 text-green-800`}>
            <CheckCircle2 className="w-3 h-3 mr-1" />
            Approved
          </span>
        );
      case 'rejected':
        return (
          <span className={`${baseClasses} bg-red-100 text-red-800`}>
            <AlertCircle className="w-3 h-3 mr-1" />
            Rejected
          </span>
        );
      case 'pending':
      default:
        return (
          <span className={`${baseClasses} bg-yellow-100 text-yellow-800`}>
            <Clock className="w-3 h-3 mr-1" />
            Pending
          </span>
        );
    }
  };

  // Filter excuses based on search term
  const filteredExcuses = useMemo(() => {
    if (!excuses?.length) return [];
    const searchLower = searchTerm.trim().toLowerCase();
    let filtered = excuses.filter((excuse) => {
      if (!searchLower) return true;
      const name = `${excuse.students?.firstname || ''} ${excuse.students?.surname || ''}`.toLowerCase();
      const id = excuse.students?.student_id?.toLowerCase() || '';
      return name.includes(searchLower) || id.includes(searchLower);
    });
    // Sort
    const dir = sortDir === 'asc' ? 1 : -1;
    filtered.sort((a, b) => {
      const nameA = `${a.students?.firstname || ''} ${a.students?.surname || ''}`.trim().toLowerCase();
      const nameB = `${b.students?.firstname || ''} ${b.students?.surname || ''}`.trim().toLowerCase();
      switch (sortKey) {
        case 'name':
          return nameA.localeCompare(nameB) * dir;
        case 'id':
          const ida = a.students?.student_id || '';
          const idb = b.students?.student_id || '';
          const na = Number(ida), nb = Number(idb);
          if (!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * dir;
          return ida.localeCompare(idb) * dir;
        case 'date':
          return (new Date(a.absence_date).getTime() - new Date(b.absence_date).getTime()) * dir;
        case 'status':
          return (a.status || '').localeCompare(b.status || '') * dir;
        default:
          return 0;
      }
    });
    return filtered;
  }, [excuses, searchTerm, sortKey, sortDir]);
  const handleSort = (key: ExcuseSortKey) => {
    if (sortKey === key) setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(key); setSortDir('asc'); }
  };

  return (
    <div className="px-6 py-4">
      <div className="mb-3">
        <div>
          <h1 className="text-2xl font-bold text-education-navy">EXCUSE APPLICATIONS</h1>
        </div>
      </div>
      
      {/* Big space between page title and card */}
      <div className="mb-16"></div>
      
      {/* Excuse Applications Section */}
      <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-semibold text-education-navy">List of Excuse Applications</h3>
          <Button 
            variant="default"
            size="sm"
            className="h-8"
            onClick={() => setIsFormOpen(true)}
          >
            <Plus className="w-4 h-4 mr-1" />
            New Application
          </Button>
        </div>
        
        {/* Big space below List of Excuse Applications label */}
        <div className="mb-8"></div>
        
        {/* Top controls row */}
        <div className="flex items-center justify-between gap-4 p-0 mb-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Showed:</span>
            <Select
              value={displayPageSize >= 999999 ? "all" : displayPageSize.toString()}
              onValueChange={(value) => {
                if (value === "all") {
                  handlePageSizeChange(999999);
                } else {
                  handlePageSizeChange(parseInt(value));
                }
              }}
            >
              <SelectTrigger className="h-8 w-24">
                <SelectValue>
                  {displayPageSize.toString()}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10</SelectItem>
                <SelectItem value="100">100</SelectItem>
                <SelectItem value="250">250</SelectItem>
                <SelectItem value="all">ALL</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Search:</span>
            <div className="relative min-w-[240px] max-w-[340px]">
              <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search applications..."
                className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                type="search"
              />
            </div>
          </div>
        </div>
        
        {/* Table View */}
        <div className="border-t border-gray-200 overflow-hidden min-h-[378px]">
          <table className="min-w-full divide-y divide-gray-200 border-b border-gray-200">
            <thead className="bg-gray-50">
              <tr className="text-xs text-black h-8">
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">Name
                    <button type="button" onClick={() => handleSort('name')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'name' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5 text-black"/> : <ChevronsDown className="w-3.5 h-3.5 text-black"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40 text-black"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">ID
                    <button type="button" onClick={() => handleSort('id')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'id' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">Date of Session
                    <button type="button" onClick={() => handleSort('date')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'date' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">Documentation</th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">Status
                    <button type="button" onClick={() => handleSort('status')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'status' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase"></th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
              {loading ? null : filteredExcuses.length === 0 ? (
                <tr className="h-8">
                  <td colSpan={6} className="px-3 py-1 text-center text-sm text-gray-500">
                    {excuses.length === 0 
                      ? 'No excuse applications found. Add your first application!'
                      : 'No applications match the current search. Try adjusting your search.'}
                  </td>
                </tr>
              ) : (
                <>
                  {filteredExcuses.map((excuse) => (
                    <tr key={excuse.id} className="hover:bg-gray-50 h-8">
                      <td className="px-3 py-1 whitespace-nowrap">
                        <div className="font-medium">
                          {excuse.students?.firstname} {excuse.students?.surname}
                        </div>
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap">
                        {excuse.students?.student_id}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap">
                        {format(new Date(excuse.absence_date), 'MMM d, yyyy')}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap">
                        {excuse.documentation_url ? (
                          <div className="w-12 h-8 bg-gray-100 rounded border overflow-hidden">
                            <img 
                              src={excuse.documentation_url} 
                              alt="Excuse letter preview" 
                              className="w-full h-full object-cover"
                            />
                          </div>
                        ) : (
                          <span className="text-sm text-gray-400">No attachment</span>
                        )}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap">
                        {getStatusDisplay(excuse.status)}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap text-right">
                        <div className="flex gap-1 justify-end">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 w-6 p-0 transition-all duration-200 hover:scale-105"
                            onClick={() => {
                              setSelectedExcuse(excuse);
                              setViewMode('view');
                              setIsViewOpen(true);
                            }}
                          >
                            <List className="h-3 w-3 text-green-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 w-6 p-0 transition-all duration-200 hover:scale-105"
                            onClick={() => {
                              setEditingExcuse(excuse);
                              setIsFormOpen(true);
                            }}
                          >
                            <Edit className="h-3 w-3 text-yellow-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 w-6 p-0 transition-all duration-200 hover:scale-105"
                            onClick={() => {
                              setDeleteTarget(excuse.id);
                              setShowDeleteConfirm(true);
                            }}
                          >
                            <Trash2 className="h-3 w-3 text-red-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Form Dialog */}
      <Dialog open={isFormOpen} onOpenChange={handleOpenChange}>
        <DialogContent className="max-w-[96vw] w-[96vw] max-h-[90vh] p-4">
          <ExcuseForm
            initialData={editingExcuse ? {
              id: editingExcuse.id,
              student_id: editingExcuse.student_id?.toString(),
              session_id: editingExcuse.session_id?.toString(),
              absence_date: editingExcuse.absence_date,
              documentation_url: editingExcuse.documentation_url
            } : undefined}
            onSuccess={handleFormSuccess}
            onCancel={handleClose}
            markAsChanged={markAsChanged}
          />
        </DialogContent>
      </Dialog>

      <UnsavedChangesDialog
        open={showConfirmDialog}
        onConfirm={confirmClose}
        onCancel={cancelClose}
      />

      {/* View Dialog */}
      <Dialog open={isViewOpen} onOpenChange={(open) => {
        setIsViewOpen(open);
        if (!open) {
          setImageZoom(1);
          setImagePan({ x: 0, y: 0 });
          setIsDragging(false);
        }
      }}>
        <DialogContent className="max-w-[96vw] w-[96vw] max-h-[90vh] p-4">
          {selectedExcuse && (
            <div className="w-full flex flex-col" style={{ height: '680px' }}>
              {/* Header */}
              <div className="pb-2 mb-3 flex-shrink-0">
                <h2 className="text-education-navy text-xl font-semibold">
                  Excuse Application Details
                </h2>
              </div>

              {/* Main Content */}
              <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                <div className="grid grid-cols-[400px_1px_1fr] gap-0 overflow-hidden" style={{ height: 'calc(100% - 50px)' }}>
                  {/* Left Column - Application Info (Read-only) */}
                  <div className="pr-6 space-y-3 overflow-y-auto">
                    {/* Student Name */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Student Name</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {selectedExcuse.students?.firstname || 'Unknown'} {selectedExcuse.students?.surname || 'Student'}
                      </div>
                    </div>

                    {/* Student ID */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Student ID</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {selectedExcuse.students?.student_id || 'N/A'}
                      </div>
                    </div>

                    {/* Program */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Program</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {selectedExcuse.students?.program || 'N/A'}
                      </div>
                    </div>

                    {/* Year & Section */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1.5">
                        <Label className="text-sm">Year</Label>
                        <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                          {selectedExcuse.students?.year || 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Session */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Session</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {selectedExcuse.sessions?.title || 'N/A'}
                      </div>
                    </div>

                    {/* Absence Date */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Absence Date</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {format(new Date(selectedExcuse.absence_date), 'MMM d, yyyy')}
                      </div>
                    </div>

                    {/* Status */}
                    <div className="space-y-1.5">
                      <Label className="text-sm">Status</Label>
                      <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                        {getStatusBadge(selectedExcuse.status)}
                      </div>
                    </div>

                    {/* Review Notes */}
                    {selectedExcuse.review_notes && (
                      <div className="space-y-1.5">
                        <Label className="text-sm">Review Notes</Label>
                        <div className="px-3 py-2 text-sm bg-gray-100 rounded-md border border-input min-h-[100px]">
                          {selectedExcuse.review_notes}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Vertical Divider */}
                  <div className="bg-gray-200 w-px"></div>

                  {/* Right Column - Image Preview */}
                  <div className="pl-6 flex flex-col min-h-0 overflow-hidden">
                    <h3 className="font-semibold text-sm mb-3 flex-shrink-0">Excuse Letter</h3>
                    
                    <div className="border rounded-lg flex-1 min-h-0 overflow-hidden bg-gray-50 relative">
                      {selectedExcuse.documentation_url ? (
                        <div className="h-full relative group">
                          <div 
                            className="h-full overflow-hidden cursor-grab active:cursor-grabbing"
                            onMouseDown={handleImageMouseDown}
                            onMouseMove={handleImageMouseMove}
                            onMouseUp={handleImageMouseUp}
                            onMouseLeave={handleImageMouseUp}
                          >
                            <img 
                              src={selectedExcuse.documentation_url} 
                              alt="Excuse letter" 
                              className="transition-transform duration-200 max-w-none object-contain w-full h-full"
                              style={{ 
                                transform: `scale(${imageZoom}) translate(${imagePan.x}px, ${imagePan.y}px)`,
                                transformOrigin: 'center center'
                              }}
                              draggable={false}
                            />
                          </div>
                          
                          {/* Zoom controls */}
                          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleZoomChange(Math.max(0.5, imageZoom - 0.25))}
                              className="h-8 w-8 p-0 bg-black/80 hover:bg-black/90 border-black/80 text-white"
                            >
                              <ZoomOut className="h-4 w-4" />
                            </Button>
                            <span className="text-xs text-white bg-black/80 px-2 py-1 rounded text-center min-w-[3rem]">
                              {Math.round(imageZoom * 100)}%
                            </span>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleZoomChange(Math.min(3, imageZoom + 0.25))}
                              className="h-8 w-8 p-0 bg-black/80 hover:bg-black/90 border-black/80 text-white"
                            >
                              <ZoomIn className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="h-full flex items-center justify-center">
                          <div className="text-center text-gray-400">
                            <FileImage className="h-16 w-16 mx-auto mb-2 opacity-50" />
                            <p className="text-sm">No excuse letter attached</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                {selectedExcuse.status === 'pending' && (
                  <div className="pt-1 flex justify-end flex-shrink-0 gap-2">
                    <Button
                      variant="outline"
                      className="text-green-600 border-green-200 hover:bg-green-50"
                      onClick={() => handleUpdateStatus(selectedExcuse.id, 'approved')}
                    >
                      <Check className="h-4 w-4 mr-2" />
                      Approve
                    </Button>
                    <Button
                      variant="outline"
                      className="text-red-600 border-red-200 hover:bg-red-50"
                      onClick={() => handleUpdateStatus(selectedExcuse.id, 'rejected')}
                    >
                      <X className="h-4 w-4 mr-2" />
                      Reject
                    </Button>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Excuse Application</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this excuse application? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={() => deleteTarget && handleDeleteExcuse(deleteTarget)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

    </div>
  );
};

const ExcuseApplication = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <ExcuseApplicationContent />
      </PageWrapper>
    </Layout>
  );
};

export default ExcuseApplication;