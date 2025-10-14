import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { CheckCircle, XCircle, Trash2, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import type { Student } from '@/types';

interface ParsedFolder {
  folderName: string;
  student_id: string;
  firstname: string;
  surname: string;
  files: File[];
  isValid: boolean;
  errorMessage?: string;
  matchedStudent?: Student;
}

interface BatchUploadProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (validFolders: ParsedFolder[]) => void;
  students: Student[];
}

const BatchUpload: React.FC<BatchUploadProps> = ({
  open,
  onOpenChange,
  onConfirm,
  students,
}) => {
  const [parsedFolders, setParsedFolders] = useState<ParsedFolder[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentFolder, setCurrentFolder] = useState('');

  const handleFolderSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsProcessing(true);
    setUploadProgress(0);

    try {
      const folderMap = new Map<string, File[]>();

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const pathParts = file.webkitRelativePath.split('/');
        
        if (pathParts.length < 2) continue;
        
        const folderName = pathParts[0];
        
        if (!folderMap.has(folderName)) {
          folderMap.set(folderName, []);
        }
        
        folderMap.get(folderName)!.push(file);
      }

      const parsed: ParsedFolder[] = [];
      const totalFolders = folderMap.size;
      let processedFolders = 0;

      for (const [folderName, folderFiles] of folderMap.entries()) {
        setCurrentFolder(folderName);
        
        const existingFolder = parsedFolders.find(f => f.folderName === folderName);
        if (!existingFolder) {
          const parsedFolder = parseFolderName(folderName, folderFiles, students);
          parsed.push(parsedFolder);
        }
        
        processedFolders++;
        setUploadProgress((processedFolders / totalFolders) * 100);
      }

      setParsedFolders(prev => [...prev, ...parsed]);
    } catch (error) {
      console.error('Error processing folders:', error);
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
      setCurrentFolder('');
      event.target.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const items = Array.from(e.dataTransfer.items);
    if (items.length === 0) return;

    setIsProcessing(true);
    setUploadProgress(0);

    try {
      const folderMap = new Map<string, File[]>();
      const processPromises: Promise<void>[] = [];

      for (const item of items) {
        if (item.kind === 'file') {
          const entry = item.webkitGetAsEntry();
          if (entry && entry.isDirectory) {
            processPromises.push(processDirectory(entry as FileSystemDirectoryEntry, folderMap));
          }
        }
      }

      await Promise.all(processPromises);

      const parsed: ParsedFolder[] = [];
      const totalFolders = folderMap.size;
      let processedFolders = 0;

      for (const [folderName, folderFiles] of folderMap.entries()) {
        setCurrentFolder(folderName);
        
        const existingFolder = parsedFolders.find(f => f.folderName === folderName);
        if (!existingFolder) {
          const parsedFolder = parseFolderName(folderName, folderFiles, students);
          parsed.push(parsedFolder);
        }
        
        processedFolders++;
        setUploadProgress((processedFolders / totalFolders) * 100);
      }

      setParsedFolders(prev => [...prev, ...parsed]);
    } catch (error) {
      console.error('Error processing dropped folders:', error);
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
      setCurrentFolder('');
    }
  };

  const processDirectory = async (
    dirEntry: FileSystemDirectoryEntry,
    folderMap: Map<string, File[]>
  ): Promise<void> => {
    return new Promise((resolve, reject) => {
      const reader = dirEntry.createReader();
      const folderName = dirEntry.name;

      const readEntries = () => {
        reader.readEntries(async (entries) => {
          if (entries.length === 0) {
            resolve();
            return;
          }

          const filePromises: Promise<void>[] = [];

          for (const entry of entries) {
            if (entry.isFile) {
              const filePromise = new Promise<void>((resolveFile) => {
                const fileEntry = entry as FileSystemFileEntry;
                fileEntry.file((file) => {
                  if (file.type.startsWith('image/')) {
                    if (!folderMap.has(folderName)) {
                      folderMap.set(folderName, []);
                    }
                    folderMap.get(folderName)!.push(file);
                  }
                  resolveFile();
                }, () => resolveFile());
              });
              filePromises.push(filePromise);
            }
          }

          await Promise.all(filePromises);
          readEntries();
        }, (error) => {
          console.error('Error reading directory entries:', error);
          reject(error);
        });
      };

      readEntries();
    });
  };

  const parseFolderName = (
    folderName: string,
    files: File[],
    students: Student[]
  ): ParsedFolder => {
    const parts = folderName.split(' - ');
    
    if (parts.length !== 2) {
      return {
        folderName,
        student_id: '',
        firstname: '',
        surname: '',
        files,
        isValid: false,
        errorMessage: 'Invalid folder name format. Expected: "student_id - firstname surname"',
      };
    }

    const student_id = parts[0].trim();
    const fullName = parts[1].trim();
    
    if (!fullName) {
      return {
        folderName,
        student_id,
        firstname: '',
        surname: '',
        files,
        isValid: false,
        errorMessage: 'Invalid name format. Name cannot be empty',
      };
    }

    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
      return {
        folderName,
        student_id: student_id,
        firstname: '',
        surname: '',
        files: imageFiles,
        isValid: false,
        errorMessage: 'No image files found in folder',
      };
    }

    if (imageFiles.length !== files.length) {
      return {
        folderName,
        student_id: student_id,
        firstname: '',
        surname: '',
        files: imageFiles,
        isValid: false,
        errorMessage: `Found ${files.length - imageFiles.length} non-image file(s). Only images allowed.`,
      };
    }

    // First, try to find student by student_id
    const matchedStudent = students.find(s => s.student_id === student_id);

    if (!matchedStudent) {
      return {
        folderName,
        student_id,
        firstname: '',
        surname: '',
        files: imageFiles,
        isValid: false,
        errorMessage: `Student ID "${student_id}" not found in database`,
      };
    }

    // Verify the name matches (compare full name to allow for multi-word first/last names)
    const expectedFullName = `${matchedStudent.firstname} ${matchedStudent.surname}`.toLowerCase();
    const providedFullName = fullName.toLowerCase();

    if (expectedFullName !== providedFullName) {
      return {
        folderName,
        student_id,
        firstname: matchedStudent.firstname,
        surname: matchedStudent.surname,
        files: imageFiles,
        isValid: false,
        errorMessage: `Name mismatch. Expected: "${matchedStudent.firstname} ${matchedStudent.surname}", Got: "${fullName}"`,
      };
    }

    return {
      folderName,
      student_id,
      firstname: matchedStudent.firstname,
      surname: matchedStudent.surname,
      files: imageFiles,
      isValid: true,
      matchedStudent,
    };
  };

  const removeFolder = (folderName: string) => {
    setParsedFolders(prev => prev.filter(f => f.folderName !== folderName));
  };

  const handleConfirm = () => {
    const validFolders = parsedFolders.filter(f => f.isValid);
    
    if (validFolders.length === 0) {
      return;
    }

    onConfirm(validFolders);
    onOpenChange(false);
    setParsedFolders([]);
  };

  const validCount = parsedFolders.filter(f => f.isValid).length;
  const invalidCount = parsedFolders.filter(f => !f.isValid).length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden p-0">
        <DialogHeader className="p-6 pb-0">
          <DialogTitle className="text-lg font-semibold">
            Batch Upload - Student Signatures
          </DialogTitle>
          <DialogDescription className="text-sm text-gray-500 mt-2">
            Select folders with student signatures. Each folder must be named: "student_id - firstname surname"
          </DialogDescription>
        </DialogHeader>

        <div className="p-6 pt-4">
          {parsedFolders.length === 0 && (
            <div 
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <label className="cursor-pointer">
                <div className="space-y-2">
                  <div className="text-gray-500">
                    <AlertCircle className="w-12 h-12 mx-auto mb-2" />
                    <p className="font-medium">Select or Drop Student Folders</p>
                    <p className="text-sm">Click to browse or drag folders here</p>
                  </div>
                  <Button variant="default" asChild>
                    <span>Choose Folders</span>
                  </Button>
                </div>
                <input
                  type="file"
                  {...({ webkitdirectory: '', directory: '', multiple: true } as any)}
                  onChange={handleFolderSelect}
                  className="hidden"
                />
              </label>
            </div>
          )}

          {isProcessing && (
            <div className="text-center py-8 space-y-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
              <div>
                <p className="text-sm font-medium text-gray-700">Processing folders...</p>
                {currentFolder && (
                  <p className="text-xs text-gray-500 mt-1">{currentFolder}</p>
                )}
              </div>
              <div className="max-w-md mx-auto">
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-xs text-gray-600 mt-1">{uploadProgress.toFixed(0)}%</p>
              </div>
            </div>
          )}

          {parsedFolders.length > 0 && (
            <>
              <div className="flex items-center gap-4 mb-4">
                <Badge variant="default" className="bg-green-100 text-green-800">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  {validCount} Valid
                </Badge>
                {invalidCount > 0 && (
                  <Badge variant="destructive">
                    <XCircle className="w-3 h-3 mr-1" />
                    {invalidCount} Invalid
                  </Badge>
                )}
                <label className="ml-auto cursor-pointer">
                  <Button variant="outline" size="sm" asChild>
                    <span>Add More Folders</span>
                  </Button>
                  <input
                    type="file"
                    {...({ webkitdirectory: '', directory: '', multiple: true } as any)}
                    onChange={handleFolderSelect}
                    className="hidden"
                  />
                </label>
              </div>

              <ScrollArea 
                className="h-[400px] hide-scrollbar"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="space-y-2">
                  {parsedFolders.map((folder, index) => (
                    <div
                      key={index}
                      className={`w-full border rounded-lg p-3 ${
                        folder.isValid
                          ? 'bg-green-50 border-green-200'
                          : 'bg-red-50 border-red-200'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            {folder.isValid ? (
                              <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                            )}
                            <div className="flex-1 min-w-0">
                              <p className="font-semibold text-sm text-gray-900 truncate">
                                {folder.folderName}
                              </p>
                              {folder.isValid && (
                                <p className="text-xs text-gray-600">
                                  {folder.files.length} image{folder.files.length !== 1 ? 's' : ''} found
                                </p>
                              )}
                            </div>
                          </div>

                          {!folder.isValid && folder.errorMessage && (
                            <div className="ml-6 text-xs text-red-600">
                              {folder.errorMessage}
                            </div>
                          )}

                          {folder.isValid && folder.matchedStudent && (
                            <div className="ml-6 text-xs text-gray-600">
                              Matched: {folder.matchedStudent.program} {folder.matchedStudent.year}-{folder.matchedStudent.section}
                            </div>
                          )}
                        </div>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFolder(folder.folderName)}
                          className="text-gray-500 hover:text-red-600 h-7 w-7 p-0 ml-2 flex-shrink-0"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>

              <div className="flex gap-3 mt-4">
                <Button
                  variant="outline"
                  onClick={() => {
                    onOpenChange(false);
                    setParsedFolders([]);
                  }}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirm}
                  disabled={validCount === 0}
                  className="flex-1"
                >
                  Add {validCount} Student{validCount !== 1 ? 's' : ''}
                </Button>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default BatchUpload;