import React, { useState, useEffect, useCallback, FC } from 'react';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Checkbox } from '@/components/ui/checkbox';
import { Search, User, Check, Eye, EyeOff } from 'lucide-react';
import { fetchStudents } from '@/lib/supabaseService';
import { formatStudentDisplay } from '@/lib/utils';
import type { Student } from '@/types';

interface StudentSelectionModalProps {
  selectedStudent?: Student | null;
  selectedStudents?: Student[];
  excludeStudents?: Student[];
  onStudentSelect?: (student: Student | null) => void;
  onStudentsSelect?: (students: Student[]) => void;
  trigger?: React.ReactNode;
  onOpenChange?: (open: boolean) => void;
  mode?: 'single' | 'multiple';
  selectionContext?: 'classCard' | 'mainButton';
}

  const StudentSelectionModal: FC<StudentSelectionModalProps> = ({
    selectedStudent,
    selectedStudents,
    excludeStudents,
    onStudentSelect,
    onStudentsSelect,
    trigger,
    isOpen,
    onOpenChange,
    mode,
    selectionContext
  }) => {
    // Auto-determine mode based on context if not explicitly set
    const effectiveMode = mode || (selectionContext === 'classCard' ? 'single' : 'multiple');
    
    // Local format function for modal display with full details
    const formatStudentDisplay = (student: Student): string => {
      return `${student.firstname} ${student.surname}\n${student.student_id} ${student.program} ${student.year} ${student.section}`;
    };
    
    const [internalOpen, setInternalOpen] = useState(false);
    const [students, setStudents] = useState<Student[]>([]);
    const [filteredStudents, setFilteredStudents] = useState<Student[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedMultiple, setSelectedMultiple] = useState<Student[]>(selectedStudents || []);
    const [displayedCount, setDisplayedCount] = useState(25);
    const [showSelected, setShowSelected] = useState(false);

    const isControlled = isOpen !== undefined && onOpenChange !== undefined;
    const open = isControlled ? isOpen : internalOpen;
    const setOpen = isControlled ? onOpenChange : setInternalOpen;

    const loadStudents = useCallback(async () => {
    try {
      setIsLoading(true);
      const studentsData = await fetchStudents();
      setStudents(studentsData);
      
      // Filter out excluded students
      const excludeStudentIds = excludeStudents.map(student => student.student_id);
      const availableStudents = studentsData.filter(student => !excludeStudentIds.includes(student.student_id));
      setFilteredStudents(availableStudents);
    } catch (error) {
      console.error('Error loading students:', error);
    } finally {
      setIsLoading(false);
    }
  }, [excludeStudents]);
  useEffect(() => {
    loadStudents();
  }, [loadStudents]);

  useEffect(() => {
    // Reset showSelected when modal opens
    setShowSelected(false);
  }, [open]);

  useEffect(() => {
    // Filter out excluded students
    const excludeStudentIds = excludeStudents.map(student => student.student_id);
    const availableStudents = students.filter(student => !excludeStudentIds.includes(student.student_id));
    
    // Removed showSelected logic - students are only filtered by excludeStudents
    
    if (!searchTerm.trim()) {
      setFilteredStudents(availableStudents);
      return;
    }

    const searchLower = searchTerm.toLowerCase();
    const filtered = availableStudents.filter(student => {
      const studentId = student.student_id.toString().toLowerCase();
      const fullName = `${student.firstname} ${student.surname}`.toLowerCase();
      return studentId.includes(searchLower) || fullName.includes(searchLower);
    });

    setFilteredStudents(filtered);
  }, [searchTerm, students, excludeStudents, effectiveMode, selectedMultiple, showSelected]);

  const handleStudentSelect = (student: Student) => {
    onStudentSelect(student);
    setOpen(false);
    setSearchTerm('');
  };

  const handleClearSelection = () => {
    if (effectiveMode === 'single' && onStudentSelect) {
      onStudentSelect(null);
    } else if (effectiveMode === 'multiple' && onStudentsSelect) {
      onStudentsSelect([]);
    }
    setOpen(false);
    setSearchTerm('');
  };

  const handleMultipleSelect = (student: Student, checked: boolean) => {
    if (checked) {
      const newSelected = [...selectedMultiple, student];
      setSelectedMultiple(newSelected);
    } else {
      const newSelected = selectedMultiple.filter(s => s.student_id !== student.student_id);
      setSelectedMultiple(newSelected);
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedMultiple(filteredStudents.slice(0, displayedCount));
    } else {
      setSelectedMultiple([]);
    }
  };

  const handleConfirmMultiple = () => {
    if (onStudentsSelect) {
      onStudentsSelect(selectedMultiple);
    }
    setOpen(false);
    setSearchTerm('');
  };

  const handleSeeMore = () => {
    setDisplayedCount(prev => prev + 25);
  };

  const defaultTrigger = (
    <Button 
      className="w-full justify-start text-left font-normal h-auto p-3"
    >
      <div className="flex items-center gap-2 w-full">
        <User className="w-4 h-4 text-gray-500" />
        {selectedStudent ? (
          <div className="flex-1">
            {(() => {
              const display = formatStudentDisplay(selectedStudent).split('\n');
              return (
                <>
                  <div className="font-bold">{display[0]}</div>
                  <div className="text-sm text-gray-500 mt-1">{display[1]}</div>
                </>
              );
            })()}
            <div className="text-sm text-gray-500 mt-1">Click to change student</div>
          </div>
        ) : (
          <div className="text-gray-500">Select a student...</div>
        )}
      </div>
    </Button>
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden p-0">
        <DialogHeader className="p-6 pb-0">
          <DialogTitle className="text-lg font-semibold">
            {effectiveMode === 'single' ? 'Select Student' : 'Add Students'}
          </DialogTitle>
        </DialogHeader>

        <div className="p-6 pt-0">
          {/* Search Bar */}
          <div className="flex gap-2 mb-4">
            {effectiveMode === 'multiple' && (
              <div className="flex items-center gap-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="select-all"
                    checked={filteredStudents.slice(0, displayedCount).length > 0 && filteredStudents.slice(0, displayedCount).every(student => selectedMultiple.some(s => s.student_id === student.student_id))}
                    onCheckedChange={(checked) => handleSelectAll(checked as boolean)}
                  />
                  <label htmlFor="select-all" className="text-sm font-medium cursor-pointer">
                    Select All
                  </label>
                </div>
                
              </div>
            )}
            
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search students..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {/* Selected Info */}
          {effectiveMode === 'single' && selectedStudent && (
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-blue-600" />
                <div>
                  {(() => {
                    const display = formatStudentDisplay(selectedStudent).split('\n');
                    return (
                      <>
                        <div className="font-bold text-blue-900">{display[0]}</div>
                        <div className="text-xs text-blue-700 mt-1">{display[1]}</div>
                      </>
                    );
                  })()}
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="text-blue-600 border-blue-300 hover:bg-blue-100"
              >
                Clear
              </Button>
            </div>
          )}

          <div className="max-h-96 overflow-y-auto hide-scrollbar">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : filteredStudents.length === 0 ? (
              <div className="text-center py-8">
                <User className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <h3 className="text-lg font-medium text-gray-500 mb-2">
                  {searchTerm ? 'No students found' : 'No students available'}
                </h3>
                <p className="text-gray-400">
                  {searchTerm 
                    ? 'Try adjusting your search terms'
                    : 'Add students to the database first'
                  }
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                {filteredStudents.slice(0, displayedCount).map((student) => (
                  <div
                    key={student.student_id}
                    className={`flex items-center pr-2 pl-0 rounded-lg transition-colors hover:bg-gray-50 ${
                      effectiveMode === 'single' && selectedStudent?.student_id === student.student_id
                        ? 'bg-blue-50'
                        : effectiveMode === 'multiple' && selectedMultiple.some(s => s.student_id === student.student_id)
                        ? 'bg-green-50'
                        : ''
                    }`}
                  >
                    {effectiveMode === 'multiple' && (
                      <Checkbox
                        checked={selectedMultiple.some(s => s.student_id === student.student_id)}
                        onCheckedChange={(checked) => handleMultipleSelect(student, checked as boolean)}
                        className="mr-2"
                      />
                    )}
                    <div 
                      onClick={() => {
                        if (effectiveMode === 'single') {
                          handleStudentSelect(student);
                        } else {
                          const isSelected = selectedMultiple.some(s => s.student_id === student.student_id);
                          handleMultipleSelect(student, !isSelected);
                        }
                      }}
                      className="flex-1 text-left cursor-pointer">
                        {(() => {
                          const display = formatStudentDisplay(student).split('\n');
                          return (
                            <>
                              <div className="font-bold text-gray-900 text-sm">{display[0]}</div>
                              <div className="text-xs text-gray-500 mt-1">{display[1]}</div>
                            </>
                          );
                        })()}
                      </div>
                    {effectiveMode === 'single' && selectedStudent?.student_id === student.student_id && (
                      <Check className="w-4 h-4 text-blue-600" />
                    )}
                  </div>
                ))}
                
                {/* See More Button */}
                {displayedCount < filteredStudents.length && (
                  <div className="pt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleSeeMore}
                      className="w-full"
                    >
                      See More ({filteredStudents.length - displayedCount} remaining)
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action Buttons */}
          {effectiveMode === 'multiple' && (
            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                onClick={() => setOpen(false)}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button
                onClick={handleConfirmMultiple}
                disabled={selectedMultiple.length === 0}
                className="flex-1"
              >
                Add {selectedMultiple.length} Student{selectedMultiple.length !== 1 ? 's' : ''}
              </Button>
            </div>
          )}
          
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default StudentSelectionModal;
