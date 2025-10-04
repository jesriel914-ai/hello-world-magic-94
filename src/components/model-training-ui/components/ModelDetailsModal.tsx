//filepath: src\components\model-training-ui\components\ModelDetailsModal.tsx
import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Brain, 
  Calendar, 
  Target, 
  Users, 
  FileText, 
  X,
  Trash2,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

type TrainedModel = {
  id: string | number;
  student_name?: string;
  student?: { id?: number; student_id?: string; firstname?: string; surname?: string; full_name?: string };
  student_full_name?: string;
  model_path?: string;
  model?: { path?: string };
  artifact_path?: string;
  training_date?: string;
  created_at?: string;
  accuracy?: number;
  training_metrics?: {
    model_type?: string;
    final_accuracy?: number;
    final_loss?: number;
    epochs_trained?: number;
    val_accuracy?: number;
    val_loss?: number;
    students?: Array<{
      id?: number;
      student_id?: string;
      firstname?: string;
      surname?: string;
      full_name?: string;
    }>;
  };
  status?: string;
  sample_count?: number;
  genuine_count?: number;
  forged_count?: number;
  student_count?: number;
  far?: number;
  frr?: number;
};

interface ModelDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  model: TrainedModel | null;
  onDelete?: (modelId: string | number) => void;
  isDeleting?: boolean;
}

const ModelDetailsModal: React.FC<ModelDetailsModalProps> = ({
  isOpen,
  onClose,
  model,
  onDelete,
  isDeleting = false
}) => {
  if (!model) return null;

  const isGlobalModel = model.student_name === 'Global Model' || 
                       model.training_metrics?.model_type === 'global_multi_student';

  const formatDateTime = (dateString?: string) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return 'N/A';
      
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      return 'N/A';
    }
  };

  const getStatusBadge = () => {
    const status = model.status || 'completed';
    
    switch (status) {
      case 'completed':
        return (
          <Badge variant="default" className="bg-green-100 text-green-800 border-green-200">
            <CheckCircle className="w-3 h-3 mr-1" />
            Completed
          </Badge>
        );
      case 'training':
        return (
          <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
            <Brain className="w-3 h-3 mr-1 animate-pulse" />
            Training
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="destructive" className="bg-red-100 text-red-800 border-red-200">
            <AlertCircle className="w-3 h-3 mr-1" />
            Failed
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            {status}
          </Badge>
        );
    }
  };

  const getStudentList = () => {
    if (isGlobalModel && model.training_metrics?.students) {
      return model.training_metrics.students;
    }
    
    if (!isGlobalModel && model.student_name) {
      return [{
        full_name: model.student_name,
        firstname: model.student?.firstname,
        surname: model.student?.surname,
        student_id: model.student?.student_id
      }];
    }
    
    return [];
  };

  const students = getStudentList();

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Model Details
            </DialogTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Model Header */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  {isGlobalModel ? '🌐 Global Model' : '👤 Individual Model'}
                </CardTitle>
                {getStatusBadge()}
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-muted-foreground">Model ID:</span>
                  <div className="font-mono text-xs">{model.id}</div>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Created:</span>
                  <div>{formatDateTime(model.training_date || model.created_at)}</div>
                </div>
                {model.accuracy !== undefined && (
                  <div>
                    <span className="font-medium text-muted-foreground">Accuracy:</span>
                    <div className="font-semibold text-green-600">
                      {Math.round(model.accuracy * 100)}%
                    </div>
                  </div>
                )}
                {model.student_count !== undefined && (
                  <div>
                    <span className="font-medium text-muted-foreground">Students:</span>
                    <div>{model.student_count}</div>
                  </div>
                )}
              </div>
              
              {model.model_path && (
                <div>
                  <span className="font-medium text-muted-foreground text-sm">Model Path:</span>
                  <div className="font-mono text-xs bg-muted p-2 rounded mt-1 break-all">
                    {model.model_path}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Training Metrics */}
          {model.training_metrics && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Training Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {model.training_metrics.final_accuracy !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">Final Accuracy:</span>
                      <div className="font-semibold text-green-600">
                        {Math.round(model.training_metrics.final_accuracy * 100)}%
                      </div>
                    </div>
                  )}
                  {model.training_metrics.val_accuracy !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">Validation Accuracy:</span>
                      <div className="font-semibold text-blue-600">
                        {Math.round(model.training_metrics.val_accuracy * 100)}%
                      </div>
                    </div>
                  )}
                  {model.training_metrics.final_loss !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">Final Loss:</span>
                      <div className="font-semibold text-red-600">
                        {model.training_metrics.final_loss.toFixed(4)}
                      </div>
                    </div>
                  )}
                  {model.training_metrics.val_loss !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">Validation Loss:</span>
                      <div className="font-semibold text-orange-600">
                        {model.training_metrics.val_loss.toFixed(4)}
                      </div>
                    </div>
                  )}
                  {model.training_metrics.epochs_trained !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">Epochs Trained:</span>
                      <div>{model.training_metrics.epochs_trained}</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Students Information */}
          {students.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  {isGlobalModel ? 'Trained Students' : 'Student Information'}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {students.map((student, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div>
                        <div className="font-medium">
                          {student.full_name || 
                           `${student.firstname || ''} ${student.surname || ''}`.trim() || 
                           'Unknown Student'
                          }
                        </div>
                        {student.student_id && (
                          <div className="text-sm text-muted-foreground">
                            ID: {student.student_id}
                          </div>
                        )}
                      </div>
                      <Badge variant="outline" className="text-xs">
                        Student {index + 1}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Additional Statistics */}
          {(model.sample_count !== undefined || model.genuine_count !== undefined || model.forged_count !== undefined) && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Training Statistics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  {model.sample_count !== undefined && (
                    <div className="text-center">
                      <div className="font-semibold text-lg">{model.sample_count}</div>
                      <div className="text-muted-foreground">Total Samples</div>
                    </div>
                  )}
                  {model.genuine_count !== undefined && (
                    <div className="text-center">
                      <div className="font-semibold text-lg text-green-600">{model.genuine_count}</div>
                      <div className="text-muted-foreground">Genuine</div>
                    </div>
                  )}
                  {model.forged_count !== undefined && (
                    <div className="text-center">
                      <div className="font-semibold text-lg text-red-600">{model.forged_count}</div>
                      <div className="text-muted-foreground">Forged</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Performance Metrics */}
          {(model.far !== undefined || model.frr !== undefined) && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {model.far !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">False Accept Rate:</span>
                      <div className="font-semibold text-red-600">
                        {(model.far * 100).toFixed(2)}%
                      </div>
                    </div>
                  )}
                  {model.frr !== undefined && (
                    <div>
                      <span className="font-medium text-muted-foreground">False Reject Rate:</span>
                      <div className="font-semibold text-orange-600">
                        {(model.frr * 100).toFixed(2)}%
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-2 pt-4 border-t">
            {onDelete && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => onDelete(model.id)}
                disabled={isDeleting}
                className="flex items-center gap-2"
              >
                {isDeleting ? (
                  <>
                    <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    Delete Model
                  </>
                )}
              </Button>
            )}
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ModelDetailsModal;
