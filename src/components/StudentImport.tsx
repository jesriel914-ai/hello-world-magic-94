import React, { useState, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from "@/components/ui/dialog";
import { Download, FileSpreadsheet, FileText, Loader2, X, Check, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import * as XLSX from 'xlsx';
import Papa from 'papaparse';
import { supabase } from "@/lib/supabase";

interface StudentImportData {
  surname: string;
  firstname: string;
  student_id: string;
  program: string;
  year: string;
  section: string;
  sex: string;
  [key: string]: string;
}

interface StudentImportProps {
  onImportComplete?: () => void;
  onImportSuccess?: () => void;
}

const StudentImport: React.FC<StudentImportProps> = ({ onImportComplete, onImportSuccess }) => {
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [validation, setValidation] = useState<{
    total: number;
    valid: number;
    errors: string[];
  } | null>(null);
  const [previewData, setPreviewData] = useState<StudentImportData[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const requiredFields = [
    'surname',
    'firstname',
    'student_id',
    'program',
    'year',
    'section'
  ];

  type StudentRecord = Record<string, string | number | null | undefined>;

  const validateStudentData = (data: StudentRecord[]): { valid: boolean; errors: string[]; data: StudentImportData[] } => {
    const errors: string[] = [];
    const validData: StudentImportData[] = [];

    data.forEach((row, index) => {
      const rowNumber = index + 2; // +2 because of 0-based index and header row
      const rowErrors: string[] = [];
      const student: Partial<StudentImportData> = {};

      // Map and validate each field
      Object.entries(row).forEach(([key, value]) => {
        const normalizedKey = key.trim().toLowerCase().replace(/[^a-z0-9]/g, '_');
        student[normalizedKey] = String(value || '').trim();
      });

      // Check required fields
      requiredFields.forEach(field => {
        if (!student[field]) {
          rowErrors.push(`Row ${rowNumber}: Missing required field '${field}'`);
        }
      });

      // Validate year format if present
      if (student.year && !/^\d+(st|nd|rd|th)$/i.test(student.year)) {
        rowErrors.push(`Row ${rowNumber}: Invalid year format. Use format like '1st', '2nd', etc.`);
      }

      if (rowErrors.length === 0) {
        validData.push(student as StudentImportData);
      } else {
        errors.push(...rowErrors);
      }
    });

    return {
      valid: errors.length === 0,
      errors,
      data: validData
    };
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setValidation(null);
    setPreviewData([]);

    try {
      const fileType = file.name.split('.').pop()?.toLowerCase() || '';
      let data: StudentRecord[] = [];

      if (fileType === 'xlsx') {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        data = XLSX.utils.sheet_to_json(worksheet);
      } else if (fileType === 'csv') {
        const text = await file.text();
        const result = Papa.parse(text, { header: true });
data = result.data as StudentRecord[];
      } else {
        throw new Error('Unsupported file type. Please upload a .xlsx or .csv file.');
      }

      // Filter out empty rows
      data = data.filter(row => 
        Object.values(row).some(value => 
          value !== null && value !== undefined && String(value).trim() !== ''
        )
      );

      const validationResult = validateStudentData(data);
      
      setPreviewData(validationResult.data);
      setValidation({
        total: data.length,
        valid: validationResult.data.length,
        errors: validationResult.errors
      });

    } catch (error) {
      console.error('Error processing file:', error);
      toast.error('Error processing file. Please check the file format and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImport = async () => {
    if (!previewData || previewData.length === 0) {
      toast.error('No data to import');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Check if user is authenticated
      const { data: { user }, error: authError } = await supabase.auth.getUser();
      
      if (authError || !user) {
        throw new Error('You must be logged in to import students');
      }
      
      const { data: validData, errors } = validateStudentData(previewData);
      
      if (errors.length > 0) {
        setValidation({
          total: previewData.length,
          valid: validData.length,
          errors: errors.slice(0, 5) // Show first 5 errors
        });
        setIsLoading(false);
        return;
      }
      
      if (validData.length === 0) {
        toast.error('No valid data to import');
        setIsLoading(false);
        return;
      }

      // Prepare data for batch insert
      const studentsToImport = validData.map(student => ({
        ...student,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }));

      // Batch insert using Supabase with proper error handling
      const { data: insertedData, error: insertError } = await supabase
        .from('students')
        .insert(studentsToImport)
        .select()
        .select('*');

      if (insertError) {
        console.error('Supabase insert error:', insertError);
        throw new Error(insertError.message || 'Failed to import students');
      }

      if (insertedData && insertedData.length > 0) {
        const importedCount = insertedData.length;
        toast.success(`Successfully imported ${importedCount} students`);
        if (onImportSuccess) onImportSuccess();
        if (onImportComplete) onImportComplete();
        
        // Reset to simple form after successful import
        resetForm();
        setOpen(false);
      } else {
        throw new Error('No data was imported');
      }
    } catch (error) {
      console.error('Error importing students:', error);
      toast.error(`Error importing students: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFileName('');
    setValidation(null);
    setPreviewData([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const downloadTemplate = () => {
    // Define the headers based on the StudentForm fields
    const headers = [
      'surname',
      'firstname',
      'student_id',
      'program',
      'year',
      'section',
      'sex'
    ];

    // Create a new workbook
    const wb = XLSX.utils.book_new();

    // Add a worksheet with the headers
    const ws = XLSX.utils.aoa_to_sheet([headers]);

    // Add some example data
    const exampleData = [
      {
        surname: 'Doe',
        firstname: 'John',
        student_id: '2023-001',
        program: 'Computer Science',
        year: '1st',
        section: 'BSCS 1A',
        sex: 'Male'
      },
      {
        surname: 'Smith',
        firstname: 'Jane',
        student_id: '2023-002',
        program: 'Information Technology',
        year: '2nd',
        section: 'BSIT 2B',
        sex: 'Female'
      }
    ];

    // Add the example data to the worksheet
    XLSX.utils.sheet_add_json(ws, exampleData, { header: headers, skipHeader: true, origin: 'A2' });

    // Add the worksheet to the workbook
    XLSX.utils.book_append_sheet(wb, ws, 'Students');

    // Generate the Excel file
    const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });

    // Create a Blob and download link
    const blob = new Blob([excelBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'student_import_template.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button 
          variant="default"
          size="sm"
        >
          <Download className="w-4 h-4 mr-1" />
          Import
        </Button>
      </DialogTrigger>
      <DialogContent className={fileName ? "max-w-[96vw] w-[96vw] max-h-[90vh] p-4" : "max-w-xl p-4"}>
        {!fileName ? (
          // Simple upload form - shown initially
          <>
            <DialogHeader>
              <DialogTitle className="text-2xl font-bold text-education-navy">
                Import Students
              </DialogTitle>
              <DialogDescription>
                Upload an Excel or CSV file to import multiple students at once.
                Download the template file to ensure proper formatting.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-6">
              <div className="border-2 border-dashed rounded-lg p-6 text-center">
                <div className="flex flex-col items-center justify-center space-y-4">
                  <div className="p-3 rounded-full bg-accent/10">
                    <Download className="w-8 h-8 text-muted-foreground" />
                  </div>
                  
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">
                      Upload Excel (.xlsx) or CSV (.csv) file
                    </p>
                    <p className="text-xs text-muted-foreground">
                      The file should contain student data with appropriate headers. 
                      <button 
                        type="button" 
                        onClick={downloadTemplate}
                        className="text-primary hover:underline ml-1"
                      >
                        Download template
                      </button>
                    </p>
                  </div>

                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isLoading}
                  >
                    Select File
                  </Button>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".xlsx,.csv"
                    className="hidden"
                    onChange={handleFileChange}
                    disabled={isLoading}
                  />
                </div>
              </div>
            </div>
          </>
        ) : (
          // Expanded 2-column form - shown after file upload
          <div className="w-full flex flex-col" style={{ height: '660px' }}>
            {/* Header */}
            <div className="pb-2 mb-3 flex-shrink-0">
              <h2 className="text-education-navy text-xl font-semibold">
                Import Students
              </h2>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              <div className="grid grid-cols-[400px_1px_1fr] gap-0 overflow-hidden" style={{ height: 'calc(100% - 50px)' }}>
                {/* Left Column - File Info */}
                <div className="pr-6 space-y-3 overflow-y-auto">
                  {/* File Name */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">File Name</label>
                    <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                      {fileName}
                    </div>
                  </div>

                  {/* Total Records */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Total Records</label>
                    <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                      {validation ? validation.total : 0}
                    </div>
                  </div>

                  {/* Valid Records */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Valid Records</label>
                    <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                      {validation ? validation.valid : 0}
                    </div>
                  </div>

                  {/* Status */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Status</label>
                    <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                      {isLoading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin text-gray-400 mr-2" />
                          Processing...
                        </>
                      ) : validation?.errors.length === 0 ? (
                        <>
                          <Check className="h-4 w-4 text-green-600 mr-2" />
                          All valid
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-4 w-4 text-red-600 mr-2" />
                          {validation?.errors.length || 0} issues
                        </>
                      )}
                    </div>
                  </div>

                  {/* Download Template */}
                  <div className="space-y-1.5 pt-2">
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={downloadTemplate}
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Template
                    </Button>
                  </div>

                  {/* Errors Section */}
                  {validation && validation.errors.length > 0 && (
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium text-red-600">Issues Found</label>
                      <div className="bg-red-50 border border-red-200 rounded-md p-3 max-h-40 overflow-y-auto text-xs space-y-1">
                        {validation.errors.slice(0, 10).map((error, index) => (
                          <div key={index} className="text-red-600">{error}</div>
                        ))}
                        {validation.errors.length > 10 && (
                          <div className="text-red-500">
                            ...and {validation.errors.length - 10} more issues
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Vertical Divider */}
                <div className="bg-gray-200 w-px"></div>

                {/* Right Column - Preview Table */}
                <div className="pl-6 flex flex-col min-h-0 overflow-hidden">
                  {/* Header with count and search */}
                  <div className="flex items-center justify-between mb-3 flex-shrink-0">
                    <label className="text-sm font-medium">
                      Students: {previewData.length}
                    </label>
                  </div>

                  {/* Table */}
                  <div className="overflow-y-auto flex-1 visible-scrollbar">
                    <table className="w-full text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr className="border-b">
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Student ID</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Name</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Program</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Year</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Section</th>
                        </tr>
                      </thead>
                      <tbody>
                        {isLoading ? (
                          <tr>
                            <td colSpan={5} className="px-3 py-12 text-center">
                              <Loader2 className="h-8 w-8 animate-spin mx-auto text-gray-400" />
                            </td>
                          </tr>
                        ) : previewData.length === 0 ? (
                          <tr>
                            <td colSpan={5} className="px-3 py-12 text-center text-gray-500">
                              No valid records found
                            </td>
                          </tr>
                        ) : (
                          previewData.map((student, index) => (
                            <tr key={index} className="hover:bg-gray-50 border-b">
                              <td className="px-3 py-1.5 text-gray-900">{student.student_id}</td>
                              <td className="px-3 py-1.5 text-gray-600">{`${student.surname}, ${student.firstname}`}</td>
                              <td className="px-3 py-1.5 text-gray-600">{student.program}</td>
                              <td className="px-3 py-1.5 text-gray-600">{student.year}</td>
                              <td className="px-3 py-1.5 text-gray-600">{student.section}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* Fixed Buttons at Bottom Right */}
            <div className="pt-1 flex justify-end gap-2 flex-shrink-0">
              <Button
                variant="outline"
                onClick={resetForm}
                disabled={isLoading}
              >
                Reset
              </Button>
              <Button
                onClick={handleImport}
                disabled={isLoading || previewData.length === 0 || (validation && validation.errors.length > 0)}
                className="bg-education-blue hover:bg-education-blue/90"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  `Import ${previewData.length} Students`
                )}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default StudentImport;