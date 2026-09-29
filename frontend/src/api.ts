export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type User = {
  user_id: number;
  email: string;
  username: string | null;
  is_active: boolean;
  created_at: string;
};

export type AuthSession = {
  access_token: string;
  refresh_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
};

export type Project = {
  project_id: number;
  owner_id: number;
  name: string;
  task_type: string;
  description: string | null;
  created_at: string;
  updated_at: string | null;
};

export type LabelClass = {
  class_id: number;
  project_id: number;
  name: string;
  class_index: number;
  color: string;
  created_at: string;
};

export type ImageItem = {
  image_id: number;
  project_id: number;
  original_filename: string | null;
  object_key: string;
  file_hash: string | null;
  width: number | null;
  height: number | null;
  status: string;
  created_at: string;
  updated_at: string | null;
};

export type AnnotationBox = {
  annotation_id?: number;
  annotation_set_id?: number;
  class_id: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence: number | null;
  keypoints?: { keypoint_index: number; x: number; y: number; visibility: number }[];
  created_at?: string;
};

export type AnnotationSet = {
  annotation_set_id: number;
  image_id: number;
  revision: number;
  source_type: string;
  source_model_id: number | null;
  parent_annotation_set_id: number | null;
  status: string;
  created_by: number | null;
  reviewed_by: number | null;
  created_at: string;
  reviewed_at: string | null;
  annotations: AnnotationBox[];
};

export type ImageAnnotationStatus = {
  image_id: number;
  annotation_set_id: number;
  source_type: "HUMAN" | "MODEL" | "HUMAN_CORRECTED";
  status: "PENDING" | "APPROVED" | "REJECTED";
  box_count: number;
};

export type DatasetVersion = {
  version_id: number;
  project_id: number;
  version_number: number;
  preprocess_config: Record<string, unknown> | null;
  augmentation_config: Record<string, unknown> | null;
  created_by: number | null;
  created_at: string;
  items: { version_id: number; image_id: number; annotation_set_id: number; split: string }[];
};

export type TrainingRun = {
  training_run_id: number;
  project_id: number;
  version_id: number;
  model_type: string;
  base_model: string | null;
  training_config: Record<string, unknown> | null;
  status: string;
  progress_percent: number | null;
  error_message: string | null;
  can_resume: boolean;
  created_at: string;
  metrics: {
    precision: number | null;
    recall: number | null;
    map50: number | null;
    map50_95: number | null;
  }[];
};

export type ModelItem = {
  model_id: number;
  project_id: number;
  training_run_id: number;
  version: number;
  weights_object_key: string;
  precision: number | null;
  recall: number | null;
  map50: number | null;
  map50_95: number | null;
  created_at: string;
};

export type ModelPredictionBox = {
  class_id: number;
  class_index: number;
  class_name: string;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence: number;
  keypoints: { x: number; y: number; visibility: number }[];
};

export type ModelPrediction = {
  model_id: number;
  project_id: number;
  image_width: number;
  image_height: number;
  boxes: ModelPredictionBox[];
};

export type ModelBatchPrediction = {
  model_id: number;
  project_id: number;
  results: ModelPrediction[];
};

export type ModelPredictionStreamEvent = {
  index: number;
  result: ModelPrediction;
};

export type AutoLabelJob = {
  job_id: number;
  project_id: number;
  model_id: number | null;
  base_model: string | null;
  confidence_threshold: number | null;
  max_predictions_per_image: number | null;
  total_images: number | null;
  processed_images: number | null;
  status: string;
  error_message: string | null;
  items: {
    job_id: number;
    image_id: number;
    annotation_set_id: number | null;
    status: string;
  }[];
};

export type YoloImportResult = {
  imported_images: number;
  imported_annotation_sets: number;
  imported_boxes: number;
  classes: string[];
};

export type ArchiveInspection = {
  archive_type: "YOLO" | "IMAGES";
  image_count: number;
  labeled_image_count: number;
  unlabeled_image_count: number;
  class_names: string[];
};

export type ModelDownloadFormat = "pt" | "onnx" | "torchscript";

const SESSION_STORAGE_KEY = "cv-training-platform.auth-session";

function loadSession(): AuthSession | null {
  try {
    const value = window.localStorage.getItem(SESSION_STORAGE_KEY);
    return value ? JSON.parse(value) as AuthSession : null;
  } catch {
    return null;
  }
}

let activeSession: AuthSession | null = loadSession();
let refreshPromise: Promise<boolean> | null = null;
const imageBlobRequests = new Map<number, Promise<Blob>>();

export function getAuthSession() {
  return activeSession;
}

export function saveAuthSession(session: AuthSession) {
  activeSession = session;
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
}

export function clearAuthSession() {
  activeSession = null;
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

function headersFor(init?: RequestInit) {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (activeSession?.access_token) headers.set("Authorization", `Bearer ${activeSession.access_token}`);
  return headers;
}

async function refreshSession(): Promise<boolean> {
  if (refreshPromise) return refreshPromise;
  const refreshToken = activeSession?.refresh_token;
  if (!refreshToken) return false;

  refreshPromise = (async () => {
    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!response.ok) {
      if (activeSession?.refresh_token === refreshToken) clearAuthSession();
      return false;
    }
    saveAuthSession(await response.json() as AuthSession);
    return true;
  })();

  try {
    return await refreshPromise;
  } finally {
    refreshPromise = null;
  }
}

async function requestResponse(path: string, init?: RequestInit, allowRefresh = true): Promise<Response> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: headersFor(init) });
  if (response.status === 401 && allowRefresh && await refreshSession()) {
    return requestResponse(path, init, false);
  }
  return response;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await requestResponse(path, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

async function requestNdjson<T>(
  path: string,
  init: RequestInit,
  onItem: (item: T) => void | Promise<void>,
): Promise<void> {
  const response = await requestResponse(path, init);
  if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
  if (!response.body) throw new Error("Streaming response is not available");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines.filter(Boolean)) {
      await onItem(JSON.parse(line) as T);
    }
    if (done) break;
  }
  if (buffer.trim()) await onItem(JSON.parse(buffer) as T);
}

async function streamPredictions(
  streamPath: string,
  fallbackPath: string,
  files: File[],
  onResult: (event: ModelPredictionStreamEvent) => void | Promise<void>,
): Promise<void> {
  const streamForm = new FormData();
  files.forEach((file) => streamForm.append("files", file));
  try {
    await requestNdjson<ModelPredictionStreamEvent>(
      streamPath,
      { method: "POST", body: streamForm },
      onResult,
    );
  } catch {
    const fallbackForm = new FormData();
    files.forEach((file) => fallbackForm.append("files", file));
    const fallback = await request<ModelBatchPrediction>(fallbackPath, {
      method: "POST",
      body: fallbackForm,
    });
    for (const [index, result] of fallback.results.entries()) {
      await onResult({ index, result });
    }
  }
}

export const api = {
  register: (payload: { email: string; username?: string; password: string }) => request<AuthSession>("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  login: (payload: { email: string; password: string }) => request<AuthSession>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  logout: (refreshToken: string) => request<void>("/auth/logout", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken }),
  }),
  me: () => request<User>("/auth/me"),
  getImageBlob: (image: ImageItem) => {
    const existing = imageBlobRequests.get(image.image_id);
    if (existing) return existing;
    const imageRequest = (async () => {
      const response = await requestResponse(`/images/${image.image_id}/file?v=${encodeURIComponent(image.file_hash ?? image.updated_at ?? image.created_at)}`);
      if (!response.ok) throw new Error(await response.text());
      return response.blob();
    })();
    imageBlobRequests.set(image.image_id, imageRequest);
    const clearRequest = () => {
      if (imageBlobRequests.get(image.image_id) === imageRequest) imageBlobRequests.delete(image.image_id);
    };
    void imageRequest.then(clearRequest, clearRequest);
    return imageRequest;
  },
  downloadModel: async (modelId: number, format: ModelDownloadFormat = "pt") => {
    const response = await requestResponse(`/models/${modelId}/download?format=${format}`);
    if (!response.ok) throw new Error(await response.text());
    return response.blob();
  },
  predictModel: (modelId: number, file: File, confidenceThreshold: number) => {
    const form = new FormData();
    form.append("file", file);
    return request<ModelPrediction>(
      `/models/${modelId}/predict?confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      { method: "POST", body: form },
    );
  },
  predictModelBatch: (modelId: number, files: File[], confidenceThreshold: number) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<ModelBatchPrediction>(
      `/models/${modelId}/predict-batch?confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      { method: "POST", body: form },
    );
  },
  streamModelPredictions: (
    modelId: number,
    files: File[],
    confidenceThreshold: number,
    onResult: (event: ModelPredictionStreamEvent) => void | Promise<void>,
  ) => {
    return streamPredictions(
      `/models/${modelId}/predict-batch-stream?confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      `/models/${modelId}/predict-batch?confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      files,
      onResult,
    );
  },
  predictPretrainedPoseBatch: (projectId: number, files: File[], confidenceThreshold: number) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<ModelBatchPrediction>(
      `/models/pose-pretrained/predict-batch?project_id=${projectId}&confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      { method: "POST", body: form },
    );
  },
  streamPretrainedPosePredictions: (
    projectId: number,
    files: File[],
    confidenceThreshold: number,
    onResult: (event: ModelPredictionStreamEvent) => void | Promise<void>,
  ) => {
    return streamPredictions(
      `/models/pose-pretrained/predict-batch-stream?project_id=${projectId}&confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      `/models/pose-pretrained/predict-batch?project_id=${projectId}&confidence_threshold=${encodeURIComponent(confidenceThreshold)}`,
      files,
      onResult,
    );
  },
  downloadDataset: async (versionId: number) => {
    const response = await requestResponse(`/datasets/versions/${versionId}/export-yolo/download`);
    if (!response.ok) throw new Error(await response.text());
    return response.blob();
  },
  listProjects: () => request<Project[]>("/projects"),
  createProject: (payload: { name: string; description?: string; task_type: "OBJECT_DETECTION" | "POSE_ESTIMATION" }) => request<Project>("/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  deleteProject: (projectId: number) => request<void>(`/projects/${projectId}`, {
    method: "DELETE",
  }),
  listClasses: (projectId: number) => request<LabelClass[]>(`/classes/project/${projectId}`),
  createClass: (payload: { project_id: number; name: string; class_index: number; color: string }) => request<LabelClass>("/classes", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  updateClass: (classId: number, payload: { name: string; color: string }) => request<LabelClass>(`/classes/${classId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  }),
  deleteClass: (classId: number) => request<void>(`/classes/${classId}`, {
    method: "DELETE",
  }),
  listImages: (projectId: number, includeArchived = false) => request<ImageItem[]>(
    `/images/project/${projectId}${includeArchived ? "?include_archived=true" : ""}`,
  ),
  updateImagesStatus: (projectId: number, imageIds: number[], status: "UPLOADED" | "ARCHIVED") => request<{ updated_count: number; status: string }>("/images/bulk-status", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId, image_ids: imageIds, status }),
  }),
  deleteImagesPermanently: (projectId: number, imageIds: number[]) => request<{ deleted_count: number }>("/images/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId, image_ids: imageIds }),
  }),
  uploadImage: (projectId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ImageItem>(`/images/upload?project_id=${projectId}`, { method: "POST", body: form });
  },
  importYoloZip: (
    projectId: number,
    file: File,
    classNames?: string,
    maxLabeledImages?: number | null,
    classIndexes?: string,
    importRemainingImages = true,
  ) => {
    const form = new FormData();
    form.append("file", file);
    const classQuery = classNames ? `&class_names=${encodeURIComponent(classNames)}` : "";
    const limitQuery = maxLabeledImages != null ? `&max_labeled_images=${maxLabeledImages}` : "";
    const indexesQuery = classIndexes ? `&class_indexes=${encodeURIComponent(classIndexes)}` : "";
    const remainingQuery = `&import_remaining_images=${importRemainingImages}`;
    return request<YoloImportResult>(`/images/import-yolo?project_id=${projectId}${classQuery}${limitQuery}${indexesQuery}${remainingQuery}`, {
      method: "POST",
      body: form,
    });
  },
  inspectArchive: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ArchiveInspection>("/images/inspect-archive", { method: "POST", body: form });
  },
  importImageArchive: (projectId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<YoloImportResult>(`/images/import-image-archive?project_id=${projectId}`, {
      method: "POST",
      body: form,
    });
  },
  listAnnotations: (imageId: number) => request<AnnotationSet[]>(`/annotations/images/${imageId}/annotation-sets`),
  listImageAnnotationStatuses: (projectId: number) => request<ImageAnnotationStatus[]>(`/annotations/projects/${projectId}/image-statuses`),
  createAnnotationSet: (imageId: number, payload: unknown) => request<AnnotationSet>(`/annotations/images/${imageId}/annotation-sets`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  acceptAnnotation: (annotationSetId: number) => request<AnnotationSet>(`/annotations/sets/${annotationSetId}/accept`, {
    method: "POST",
    body: JSON.stringify({}),
  }),
  rejectAnnotation: (annotationSetId: number) => request<AnnotationSet>(`/annotations/sets/${annotationSetId}/reject`, {
    method: "POST",
    body: JSON.stringify({}),
  }),
  rejectAnnotationsBulk: (annotationSetIds: number[]) => request<{ rejected_count: number; skipped_count: number }>("/annotations/sets/reject-bulk", {
    method: "POST",
    body: JSON.stringify({ annotation_set_ids: annotationSetIds }),
  }),
  acceptAnnotationsBulk: (annotationSetIds: number[]) => request<{ accepted_count: number; skipped_count: number }>("/annotations/sets/accept-bulk", {
    method: "POST",
    body: JSON.stringify({ annotation_set_ids: annotationSetIds }),
  }),
  correctAnnotation: (annotationSetId: number, payload: unknown) => request<AnnotationSet>(`/annotations/sets/${annotationSetId}/correct`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  editAnnotation: (annotationSetId: number, payload: unknown) => request<AnnotationSet>(`/annotations/sets/${annotationSetId}/edit`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  listDatasets: (projectId: number) => request<DatasetVersion[]>(`/datasets/projects/${projectId}/versions`),
  createDataset: (payload: unknown) => request<DatasetVersion>("/datasets/versions", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  deleteDataset: (versionId: number) => request<void>(`/datasets/versions/${versionId}`, {
    method: "DELETE",
  }),
  buildYolo: (versionId: number) => request<{ image_count: number; label_count: number; output_dir: string }>(`/datasets/versions/${versionId}/build-yolo`, {
    method: "POST",
  }),
  exportYolo: (versionId: number) => request<{ archive_path: string }>(`/datasets/versions/${versionId}/export-yolo`, {
    method: "POST",
  }),
  listTrainingRuns: (projectId: number) => request<TrainingRun[]>(`/training-runs/project/${projectId}`),
  createTrainingRun: (payload: unknown) => request<TrainingRun>("/training-runs", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  startYoloTraining: (runId: number) => request<{ training_run: TrainingRun; model: ModelItem }>(`/training-runs/${runId}/start-local-yolo`, {
    method: "POST",
  }),
  resumeYoloTraining: (runId: number) => request<{ training_run: TrainingRun; model: ModelItem }>(`/training-runs/${runId}/resume-local-yolo`, {
    method: "POST",
  }),
  listModels: (projectId: number) => request<ModelItem[]>(`/models/project/${projectId}`),
  deleteModel: (modelId: number) => request<void>(`/models/${modelId}`, {
    method: "DELETE",
  }),
  listAutoLabelJobs: (projectId: number) => request<AutoLabelJob[]>(`/auto-label-jobs/project/${projectId}`),
  createAutoLabelJob: (payload: unknown) => request<AutoLabelJob>("/auto-label-jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  startYoloAutoLabel: (jobId: number) => request<{ job: AutoLabelJob; suggestions: AnnotationSet[] }>(`/auto-label-jobs/${jobId}/start-local-yolo`, {
    method: "POST",
  }),
  startYoloAutoLabelAsync: (jobId: number) => request<AutoLabelJob>(`/auto-label-jobs/${jobId}/start-local-yolo-async`, {
    method: "POST",
  }),
  getAutoLabelJob: (jobId: number) => request<AutoLabelJob>(`/auto-label-jobs/${jobId}`),
};
