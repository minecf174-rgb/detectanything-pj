import {
  Archive,
  ArrowLeft,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleDot,
  Copy,
  Database,
  Download,
  Eye,
  Filter,
  Image as ImageIcon,
  Keyboard,
  Layers3,
  MoreHorizontal,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Settings2,
  TestTube2,
  Trash2,
  Undo2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, FormEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent, ReactNode } from "react";
import {
  AnnotationBox,
  AnnotationSet,
  api,
  ApiHttpError,
  ArchiveInspection,
  AutoLabelJob,
  clearAuthSession,
  DatasetVersion,
  getAuthSession,
  ImageItem,
  ImageAnnotationStatus,
  LabelClass,
  ModelItem,
  ModelImportResult,
  ModelDownloadFormat,
  ModelPrediction,
  Project,
  saveAuthSession,
  TrainingRun,
  type AuthSession,
} from "./api";
import {
  DEFAULT_SHORTCUTS,
  loadShortcuts,
  matchesShortcut,
  saveShortcuts,
  shortcutConflict,
  shortcutFromEvent,
  SHORTCUT_DEFINITIONS,
  type ShortcutBindings,
} from "./shortcuts";

type DraftBox = {
  class_id: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  keypoints?: PosePoint[];
};

type BoxGeometry = Pick<DraftBox, "x_min" | "y_min" | "x_max" | "y_max">;

type PosePoint = { x: number; y: number; visibility: number };
type PoseVisibilityMode = "visible" | "occluded";

type ReviewBox = DraftBox & {
  confidence: number | null;
};

type AcceptQueueItem = { set: AnnotationSet; projectId: number; state: "queued" | "failed"; error?: string };

type ReviewHistoryItem = {
  boxes: ReviewBox[];
  selectedBoxIndex: number | null;
  selectedPointIndex: number;
};

type ImageFrame = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type TestImage = {
  key: string;
  fingerprint: string;
  file: File;
  url: string;
};

type TestRunProgress = {
  processed: number;
  total: number;
  running: boolean;
  elapsedMs: number;
  currentFrom: number;
  currentTo: number;
  currentImageIndex: number | null;
};

type OperationProgress = {
  title: string;
  detail: string;
  startedAt: number;
  completed?: number;
  total?: number;
  error?: string;
};

type TestResultFilter = "all" | "detected" | "no-detections" | "failed";

type EditMode = "move" | "nw" | "ne" | "sw" | "se";

const POSE_KEYPOINT_NAMES = [
  "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder",
  "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee",
  "right knee", "left ankle", "right ankle",
];

const COCO_SKELETON_EDGES = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
  [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
] as const;

const DEFAULT_CLASS_COLORS = [
  "#e11d48", "#f97316", "#f59e0b", "#16a34a", "#0d9488",
  "#0891b2", "#2563eb", "#7c3aed", "#c026d3", "#64748b",
] as const;

type AutoLabelResult = {
  image: ImageItem;
  annotationSet: AnnotationSet | null;
  status: string;
};

type AutoLabelResultFilter = "all" | "detected" | "empty" | "failed" | "accepted";
type ImageManagerFilter = "all" | "active" | "archived";
type WorkspaceSection = "overview" | "data" | "annotate" | "train" | "import-model" | "models" | "test";

type YoloTrainingSettings = {
  baseModel: string;
  epochs: number;
  imgsz: number;
  batch: number;
  patience: number;
  optimizer: "auto" | "SGD" | "Adam" | "AdamW";
  lr0: number;
  workers: number;
  device: "auto" | "mps" | "cpu";
  cache: boolean;
  amp: boolean;
  cos_lr: boolean;
};

const defaultTrainingSettings: YoloTrainingSettings = {
  baseModel: "yolo26s.pt",
  epochs: 50,
  imgsz: 640,
  batch: 4,
  patience: 25,
  optimizer: "auto",
  lr0: 0.01,
  workers: 0,
  device: "mps",
  cache: false,
  amp: true,
  cos_lr: false,
};

function AuthScreen({ onAuthenticated }: { onAuthenticated: (session: AuthSession) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const session = mode === "login"
        ? await api.login({ email: email.trim(), password })
        : await api.register({ email: email.trim(), username: username.trim() || undefined, password });
      onAuthenticated(session);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-brand">
          <span>CV PLATFORM</span>
          <h1 id="auth-title">Detect Anything</h1>
          <p>Prepare data, train models, and review predictions in one workspace.</p>
        </div>
        <div className="auth-tabs" role="tablist" aria-label="Authentication">
          <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")} type="button">Sign in</button>
          <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")} type="button">Create account</button>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label>Email<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
          {mode === "register" && <label>Name <span className="optional">optional</span><input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} /></label>}
          <label>Password{mode === "register" && <span className="optional">minimum 12 characters</span>}<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(event) => setPassword(event.target.value)} minLength={mode === "register" ? 12 : 1} required /></label>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="primary auth-submit" disabled={busy}>{busy ? "Please wait..." : mode === "login" ? "Sign in" : "Create account"}</button>
        </form>
      </section>
    </main>
  );
}

function ProjectDashboard({
  projects,
  busy,
  onOpenCreate,
  onOpen,
}: {
  projects: Project[];
  busy: boolean;
  onOpenCreate: () => void;
  onOpen: (projectId: number) => void;
}) {
  return (
    <main className="project-dashboard">
      <header className="dashboard-heading">
        <div>
          <span>WORKSPACE</span>
          <h2>Your projects</h2>
          <p>Create a project or open one to manage images, labels, datasets, and models.</p>
        </div>
        <button className="primary" onClick={onOpenCreate} disabled={busy}><Plus size={16} /> New project</button>
      </header>
      {projects.length === 0 ? (
        <section className="dashboard-empty">
          <ImageIcon size={28} />
          <h3>No projects yet</h3>
          <p>Create your first object-detection project to begin uploading images and labels.</p>
        </section>
      ) : (
        <section className="project-grid" aria-label="Projects">
          {[...projects].reverse().map((project) => (
            <article className="project-card" key={project.project_id}>
              <button className="project-card-main" onClick={() => onOpen(project.project_id)}>
                <span className="project-icon"><ImageIcon size={20} /></span>
                <span className="project-card-name">{project.name}</span>
                <span className="project-task">{project.task_type.replace(/_/g, " ")}</span>
                {project.description && <span className="project-description">{project.description}</span>}
                <span className="project-date">Created {new Date(project.created_at).toLocaleDateString()}</span>
              </button>
            </article>
          ))}
        </section>
      )}
    </main>
  );
}

export function App() {
  const [session, setSession] = useState<AuthSession | null>(() => getAuthSession());
  const [shortcuts, setShortcuts] = useState<ShortcutBindings>(() => loadShortcuts(getAuthSession()?.user.user_id));
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [imageAnnotationStatuses, setImageAnnotationStatuses] = useState<ImageAnnotationStatus[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationSet[]>([]);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [datasetSelectionOpen, setDatasetSelectionOpen] = useState(false);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelImportFile, setModelImportFile] = useState<File | null>(null);
  const [modelImportResult, setModelImportResult] = useState<ModelImportResult | null>(null);
  const [modelImportError, setModelImportError] = useState<string | null>(null);
  const [autoJobs, setAutoJobs] = useState<AutoLabelJob[]>([]);

  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [activeView, setActiveView] = useState<"dashboard" | "workspace">("dashboard");
  const [workspaceSection, setWorkspaceSection] = useState<WorkspaceSection>("annotate");
  const [selectedDatasetVersionId, setSelectedDatasetVersionId] = useState<number | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [classFilterId, setClassFilterId] = useState<number | null>(null);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [projectActionsOpen, setProjectActionsOpen] = useState(false);
  const [className, setClassName] = useState("object");
  const [classColor, setClassColor] = useState("#16a34a");
  const [editingClassId, setEditingClassId] = useState<number | null>(null);
  const [editingClassName, setEditingClassName] = useState("");
  const [editingClassColor, setEditingClassColor] = useState("#16a34a");
  const [importLimit, setImportLimit] = useState("");
  const [importRemainingImages, setImportRemainingImages] = useState(true);
  const [importClassMode, setImportClassMode] = useState<"all" | "selected">("all");
  const [importClassIndexes, setImportClassIndexes] = useState("");
  const [importClassNames, setImportClassNames] = useState("");
  const [pendingImportArchive, setPendingImportArchive] = useState<File | null>(null);
  const [archiveInspection, setArchiveInspection] = useState<ArchiveInspection | null>(null);
  const [imageSelectorOpen, setImageSelectorOpen] = useState(false);
  const [imageSelectorInitialView, setImageSelectorInitialView] = useState<"browse" | "negative">("browse");
  const [datasetToView, setDatasetToView] = useState<DatasetVersion | null>(null);
  const [modelDownloadFormats, setModelDownloadFormats] = useState<Record<number, ModelDownloadFormat>>({});
  const [modelActionsOpenId, setModelActionsOpenId] = useState<number | null>(null);
  const [draftBoxes, setDraftBoxes] = useState<DraftBox[]>([]);
  const [pendingClassBox, setPendingClassBox] = useState<BoxGeometry | null>(null);
  const [saveQueueRevision, setSaveQueueRevision] = useState(0);
  const [acceptQueueRevision, setAcceptQueueRevision] = useState(0);
  const [poseCollecting, setPoseCollecting] = useState(false);
  const [posePoints, setPosePoints] = useState<PosePoint[]>([]);
  const [poseVisibilityMode, setPoseVisibilityMode] = useState<PoseVisibilityMode>("visible");
  const [reviewSetId, setReviewSetId] = useState<number | null>(null);
  const [manualEditSetId, setManualEditSetId] = useState<number | null>(null);
  const [reviewBoxes, setReviewBoxes] = useState<ReviewBox[]>([]);
  const [reviewHistory, setReviewHistory] = useState<ReviewHistoryItem[]>([]);
  const [selectedReviewBoxIndex, setSelectedReviewBoxIndex] = useState<number | null>(null);
  const [selectedPoseKeypointIndex, setSelectedPoseKeypointIndex] = useState(0);
  const [poseEditVisibilityMode, setPoseEditVisibilityMode] = useState<PoseVisibilityMode>("visible");
  const [annotationImageId, setAnnotationImageId] = useState<number | null>(null);
  const [autoLabelResults, setAutoLabelResults] = useState<AutoLabelResult[] | null>(null);
  const [autoLabelResultsModelName, setAutoLabelResultsModelName] = useState("");
  const [reviewToOpen, setReviewToOpen] = useState<AnnotationSet | null>(null);
  const [trainingSetupOpen, setTrainingSetupOpen] = useState(false);
  const [trainingSettings, setTrainingSettings] = useState<YoloTrainingSettings>(defaultTrainingSettings);
  const [selectedAutoLabelModelId, setSelectedAutoLabelModelId] = useState<number | "pose-pretrained" | null>(null);
  const [autoLabelSettingsOpen, setAutoLabelSettingsOpen] = useState(false);
  const [autoLabelConfidenceInput, setAutoLabelConfidenceInput] = useState("0.25");
  const [selectedTestModelId, setSelectedTestModelId] = useState<number | "pose-pretrained" | null>(null);
  const [testImages, setTestImages] = useState<TestImage[]>([]);
  const [testDatasetSource, setTestDatasetSource] = useState<string | null>(null);
  const [testPredictions, setTestPredictions] = useState<(ModelPrediction | null)[]>([]);
  const [testFailures, setTestFailures] = useState<Record<number, string>>({});
  const [testProgress, setTestProgress] = useState<TestRunProgress>({ processed: 0, total: 0, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
  const [testConfidenceThreshold, setTestConfidenceThreshold] = useState(0.25);
  const [imageManagerOpen, setImageManagerOpen] = useState(false);
  const [managedImages, setManagedImages] = useState<ImageItem[]>([]);
  const [selectedManagedImageIds, setSelectedManagedImageIds] = useState<Set<number>>(new Set());
  const [imageManagerFilter, setImageManagerFilter] = useState<ImageManagerFilter>("all");
  const [imageManagerSearch, setImageManagerSearch] = useState("");
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const [operationProgress, setOperationProgress] = useState<OperationProgress | null>(null);
  const [annotationSaveState, setAnnotationSaveState] = useState<"idle" | "pending" | "saving" | "saved" | "error">("idle");
  const [drawingBox, setDrawingBox] = useState(false);
  const [imageSwitching, setImageSwitching] = useState(false);
  const annotationSaveTail = useRef<Promise<void>>(Promise.resolve());
  const acceptQueueTail = useRef<Promise<void>>(Promise.resolve());
  const acceptQueue = useRef(new Map<number, AcceptQueueItem>());
  const queuedBoxesByImage = useRef(new Map<number, DraftBox[]>());
  const failedBoxesByImage = useRef(new Map<number, DraftBox[]>());
  const lastQueuedDraft = useRef<{ imageId: number; boxes: DraftBox[] } | null>(null);
  const selectedImageIdRef = useRef<number | null>(null);
  const selectedProjectIdRef = useRef<number | null>(null);
  const reviewShortcutInFlight = useRef(false);
  const projectRequestId = useRef(0);
  const projectActionsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => { setShortcuts(loadShortcuts(session?.user.user_id)); }, [session?.user.user_id]);

  const updateShortcuts = (next: ShortcutBindings) => {
    setShortcuts(next);
    if (session) saveShortcuts(session.user.user_id, next);
  };

  const selectedImage = images.find((image) => image.image_id === selectedImageId) ?? null;
  const selectedProject = projects.find((project) => project.project_id === selectedProjectId) ?? null;
  selectedImageIdRef.current = selectedImageId;
  selectedProjectIdRef.current = selectedProjectId;
  const queuedImageBoxes = selectedImageId == null ? [] : queuedBoxesByImage.current.get(selectedImageId) ?? [];
  const failedImageBoxes = selectedImageId == null ? [] : failedBoxesByImage.current.get(selectedImageId) ?? [];
  const queuedImageCount = queuedBoxesByImage.current.size;
  const failedImageCount = failedBoxesByImage.current.size;
  void saveQueueRevision;
  void acceptQueueRevision;
  const queuedAcceptCount = [...acceptQueue.current.values()].filter((item) => item.state === "queued").length;
  const failedAcceptCount = acceptQueue.current.size - queuedAcceptCount;
  const reviewActionOnCurrentImage = selectedImageId != null && [...acceptQueue.current.values()]
    .some((item) => item.set.image_id === selectedImageId);
  const selectedProjectTask = selectedProject?.task_type === "POSE_ESTIMATION" ? "POSE_ESTIMATION" : "OBJECT_DETECTION";
  const poseProject = selectedProjectTask === "POSE_ESTIMATION";
  const visibleAnnotations = annotationImageId === selectedImageId ? annotations : [];
  const latestImageAnnotationSet = visibleAnnotations[visibleAnnotations.length - 1] ?? null;
  const latestApproved = [...visibleAnnotations].reverse().find((set) => set.status === "APPROVED") ?? null;
  const pendingSuggestions = visibleAnnotations.filter((set) => set.source_type === "MODEL" && set.status === "PENDING");
  const activeReviewSet = pendingSuggestions.find((set) => set.annotation_set_id === reviewSetId) ?? null;
  const manualEditSet = manualEditSetId == null
    ? null
    : visibleAnnotations.find((set) => set.annotation_set_id === manualEditSetId && set.status === "APPROVED") ?? null;
  const editableSet = activeReviewSet ?? manualEditSet;
  const editingManualLabels = activeReviewSet == null && manualEditSet != null;
  const classColorById = useMemo(
    () => Object.fromEntries(classes.map((labelClass) => [labelClass.class_id, labelClass.color])),
    [classes],
  );
  const classNameById = useMemo(
    () => Object.fromEntries(classes.map((labelClass) => [labelClass.class_id, labelClass.name])),
    [classes],
  );
  const pendingImageAnnotationStatuses = useMemo(
    () => imageAnnotationStatuses.filter((status) => status.status === "PENDING"),
    [imageAnnotationStatuses],
  );
  const pendingReviewImageIds = useMemo(
    () => new Set(pendingImageAnnotationStatuses.map((status) => status.image_id)),
    [pendingImageAnnotationStatuses],
  );
  const pendingProjectAnnotationSetIds = useMemo(
    () => pendingImageAnnotationStatuses.map((status) => status.annotation_set_id),
    [pendingImageAnnotationStatuses],
  );

  useEffect(() => {
    if (!projectActionsOpen) return;
    const closeOutside = (event: MouseEvent) => {
      if (!projectActionsRef.current?.contains(event.target as Node)) setProjectActionsOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setProjectActionsOpen(false);
    };
    document.addEventListener("mousedown", closeOutside);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOutside);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [projectActionsOpen]);

  useEffect(() => {
    if (modelActionsOpenId == null) return;
    const closeOutside = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element) || !target.closest(`[data-model-actions="${modelActionsOpenId}"]`)) {
        setModelActionsOpenId(null);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setModelActionsOpenId(null);
    };
    document.addEventListener("mousedown", closeOutside);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOutside);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [modelActionsOpenId]);
  const latestDataset = datasets[datasets.length - 1] ?? null;
  const selectedDataset = datasets.find((dataset) => dataset.version_id === selectedDatasetVersionId) ?? latestDataset;
  const latestModel = models[models.length - 1] ?? null;
  const selectedImageIndex = selectedImageId
    ? images.findIndex((image) => image.image_id === selectedImageId) + 1
    : 0;
  const classFilterImageIds = useMemo(() => new Set(
    classFilterId == null ? [] : imageAnnotationStatuses
      .filter((status) => status.class_ids.includes(classFilterId))
      .map((status) => status.image_id)
  ), [classFilterId, imageAnnotationStatuses]);
  const classFilterName = classes.find((labelClass) => labelClass.class_id === classFilterId)?.name ?? null;
  const classFilterCount = useMemo(() => images.filter((image) => classFilterImageIds.has(image.image_id)).length,
    [images, classFilterImageIds]);

  const adjacentAnnotationImage = (direction: -1 | 1) => {
    if (images.length === 0) return null;
    const currentIndex = selectedImageId == null
      ? (direction === 1 ? -1 : images.length)
      : images.findIndex((image) => image.image_id === selectedImageId);
    for (let index = currentIndex + direction; index >= 0 && index < images.length; index += direction) {
      if (classFilterId == null || classFilterImageIds.has(images[index].image_id)) return images[index];
    }
    return null;
  };

  const selectAnnotationImage = async (imageId: number) => {
    if (imageId === selectedImageId) return true;
    if (poseCollecting && posePoints.length > 0) {
      setStatus("Finish or cancel the current pose before changing images");
      return false;
    }
    if (draftBoxes.length > 0 && annotationImageId !== selectedImageId) {
      setStatus("Wait for this image's labels to load before changing images");
      return false;
    }
    setImageSwitching(true);
    try {
      if (draftBoxes.length > 0) void saveManualAnnotation();
      if (poseCollecting) {
        setPoseCollecting(false);
        setPosePoints([]);
      }
      setPendingClassBox(null);
      selectedImageIdRef.current = imageId;
      setSelectedImageId(imageId);
      return true;
    } finally {
      setImageSwitching(false);
    }
  };

  const navigateSelectedImage = (direction: -1 | 1) => {
    const nextImage = adjacentAnnotationImage(direction);
    if (nextImage) void selectAnnotationImage(nextImage.image_id);
  };

  const copyPreviousLabels = async () => {
    if (selectedImageIndex <= 1 || busy || editableSet || poseCollecting) return;
    const previousImage = images[selectedImageIndex - 2];
    if (!previousImage) return;
    setBusy(true);
    setStatus("Loading labels from previous image...");
    try {
      const previousSets = await api.listAnnotations(previousImage.image_id);
      const sourceSet = [...previousSets].reverse().find(
        (annotationSet) => annotationSet.status !== "REJECTED" && annotationSet.annotations.length > 0,
      );
      if (!sourceSet) {
        setStatus("The previous image has no labels to copy");
        return;
      }
      const copiedBoxes: DraftBox[] = sourceSet.annotations.map((box) => ({
        class_id: box.class_id,
        x_min: box.x_min,
        y_min: box.y_min,
        x_max: box.x_max,
        y_max: box.y_max,
        keypoints: box.keypoints
          ? [...box.keypoints]
            .sort((left, right) => left.keypoint_index - right.keypoint_index)
            .map(({ x, y, visibility }) => ({ x, y, visibility }))
          : undefined,
      }));
      setDraftBoxes(copiedBoxes);
      setSelectedClassId(copiedBoxes[0]?.class_id ?? selectedClassId);
      setStatus(`Copied ${copiedBoxes.length} label${copiedBoxes.length === 1 ? "" : "s"} from image ${selectedImageIndex - 1}. Adjust and save.`);
    } catch (error) {
      setStatus(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const loadProjects = useCallback(async () => {
    const nextProjects = await api.listProjects();
    setProjects(nextProjects);
    setSelectedProjectId((current) => (
      current && nextProjects.some((project) => project.project_id === current)
        ? current
        : nextProjects[0]?.project_id ?? null
    ));
    if (nextProjects.length === 0) clearProjectState();
    return nextProjects;
  }, []);

  const refreshAll = useCallback(async () => {
    setBusy(true);
    try {
      await loadProjects();
      setStatus("Loaded workspace");
    } catch (error) {
      setStatus(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }, [loadProjects]);

  const clearProjectState = () => {
    setDatasetSelectionOpen(false);
    projectRequestId.current += 1;
    setClasses([]);
    setImages([]);
    setImageAnnotationStatuses([]);
    setAnnotations([]);
    setDatasets([]);
    setSelectedDatasetVersionId(null);
    setRuns([]);
    setModels([]);
    setModelImportFile(null);
    setModelImportResult(null);
    setModelImportError(null);
    setAutoJobs([]);
    setDraftBoxes([]);
    setPendingClassBox(null);
    queuedBoxesByImage.current.clear();
    failedBoxesByImage.current.clear();
    acceptQueue.current.clear();
    lastQueuedDraft.current = null;
    setSaveQueueRevision((current) => current + 1);
    setAcceptQueueRevision((current) => current + 1);
    setImageSwitching(false);
    setPoseCollecting(false);
    setPosePoints([]);
    setPoseVisibilityMode("visible");
    setReviewSetId(null);
    setManualEditSetId(null);
    setReviewBoxes([]);
    setReviewHistory([]);
    setSelectedReviewBoxIndex(null);
    setSelectedPoseKeypointIndex(0);
    setPoseEditVisibilityMode("visible");
    setSelectedImageId(null);
    setSelectedClassId(null);
    setClassFilterId(null);
    setEditingClassId(null);
    setAnnotationImageId(null);
    setAutoLabelResults(null);
    setReviewToOpen(null);
    setTrainingSetupOpen(false);
    setSelectedAutoLabelModelId(null);
    setSelectedTestModelId(null);
    setTestImages((current) => {
      current.forEach((image) => URL.revokeObjectURL(image.url));
      return [];
    });
    setTestPredictions([]);
    setTestDatasetSource(null);
    setTestFailures({});
    setTestProgress({ processed: 0, total: 0, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
    setTestConfidenceThreshold(0.25);
    setWorkspaceSection("annotate");
    setImageManagerOpen(false);
    setImageSelectorOpen(false);
    setImageSelectorInitialView("browse");
    setPendingImportArchive(null);
    setArchiveInspection(null);
    setDatasetToView(null);
    setModelActionsOpenId(null);
    setManagedImages([]);
    setSelectedManagedImageIds(new Set());
  };

  const refreshProject = useCallback(async (projectId: number) => {
    const requestId = ++projectRequestId.current;
    setBusy(true);
    try {
      const [nextClasses, nextImages, nextImageStatuses, nextDatasets, nextRuns, nextModels, nextJobs] = await Promise.all([
        api.listClasses(projectId),
        api.listImages(projectId),
        api.listImageAnnotationStatuses(projectId),
        api.listDatasets(projectId),
        api.listTrainingRuns(projectId),
        api.listModels(projectId),
        api.listAutoLabelJobs(projectId),
      ]);
      if (requestId !== projectRequestId.current) return;
      setClasses(nextClasses);
      setImages(nextImages);
      setImageAnnotationStatuses(nextImageStatuses);
      setDatasets(nextDatasets);
      setRuns(nextRuns);
      setModels(nextModels);
      setAutoJobs(nextJobs);
      setSelectedImageId((current) => (
        current && nextImages.some((image) => image.image_id === current) ? current : nextImages[0]?.image_id ?? null
      ));
      setSelectedClassId((current) => (
        current && nextClasses.some((labelClass) => labelClass.class_id === current)
          ? current
          : nextClasses[0]?.class_id ?? null
      ));
      setStatus("Project refreshed");
    } catch (error) {
      if (requestId !== projectRequestId.current) return;
      setStatus(errorMessage(error));
    } finally {
      if (requestId === projectRequestId.current) setBusy(false);
    }
  }, []);

  const refreshAnnotations = useCallback(async (imageId: number) => {
    try {
      const nextAnnotations = await api.listAnnotations(imageId);
      if (selectedImageIdRef.current !== imageId) return;
      setAnnotations(nextAnnotations);
      setAnnotationImageId(imageId);
    } catch (error) {
      setStatus(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    if (!session) {
      setAuthChecked(true);
      return;
    }
    void api.me().then(() => {
      setAuthChecked(true);
      void refreshAll();
    }).catch((error) => {
      setAuthChecked(true);
      if (error instanceof ApiHttpError && error.status === 401) {
        clearAuthSession();
        setSession(null);
      } else {
        setStatus(errorMessage(error));
      }
    });
  }, [refreshAll, session?.access_token]);

  useEffect(() => {
    setDraftBoxes([]);
    setPendingClassBox(null);
    setAnnotationSaveState(selectedImageId != null && failedBoxesByImage.current.has(selectedImageId)
      ? "error"
      : selectedImageId != null && queuedBoxesByImage.current.has(selectedImageId) ? "saving" : "idle");
    setDrawingBox(false);
    setReviewSetId(null);
    setManualEditSetId(null);
    setReviewBoxes([]);
    setReviewHistory([]);
    setSelectedReviewBoxIndex(null);
    setSelectedPoseKeypointIndex(0);
    setAnnotations([]);
    setAnnotationImageId(null);
    if (selectedImageId) void refreshAnnotations(selectedImageId);
  }, [refreshAnnotations, selectedImageId]);

  useEffect(() => {
    if (classes.length === 0) {
      setSelectedClassId(null);
      setClassFilterId(null);
      return;
    }
    if (!selectedClassId || !classes.some((labelClass) => labelClass.class_id === selectedClassId)) {
      setSelectedClassId(classes[0].class_id);
    }
    if (classFilterId != null && !classes.some((labelClass) => labelClass.class_id === classFilterId)) {
      setClassFilterId(null);
    }
  }, [classes, selectedClassId, classFilterId]);

  useEffect(() => {
    setTrainingSettings((current) => ({
      ...current,
      baseModel: selectedProjectTask === "POSE_ESTIMATION" ? "yolo26n-pose.pt" : "yolo26s.pt",
    }));
  }, [selectedProjectId, selectedProjectTask]);

  useEffect(() => {
    if (!trainingSettings.baseModel.startsWith("model:")) return;
    const sourceModelId = Number(trainingSettings.baseModel.slice("model:".length));
    if (!models.some((model) => model.model_id === sourceModelId)) {
      setTrainingSettings((current) => ({ ...current, baseModel: "yolo26s.pt" }));
    }
  }, [models, trainingSettings.baseModel]);

  useEffect(() => {
    setSelectedAutoLabelModelId((current) => (
      (poseProject && current === "pose-pretrained") || (typeof current === "number" && models.some((model) => model.model_id === current))
        ? current
        : models[models.length - 1]?.model_id ?? (poseProject ? "pose-pretrained" : null)
    ));
  }, [models, poseProject]);

  useEffect(() => {
    setSelectedTestModelId((current) => (
      current === "pose-pretrained" || (current != null && models.some((model) => model.model_id === current))
        ? current
        : models[models.length - 1]?.model_id ?? null
    ));
  }, [models]);

  useEffect(() => {
    setSelectedDatasetVersionId((current) => (
      current && datasets.some((dataset) => dataset.version_id === current)
        ? current
        : datasets[datasets.length - 1]?.version_id ?? null
    ));
  }, [datasets]);

  useEffect(() => {
    if (!reviewToOpen || selectedImageId !== reviewToOpen.image_id) return;
    setDraftBoxes([]);
    setPendingClassBox(null);
    setAnnotations([reviewToOpen]);
    setAnnotationImageId(reviewToOpen.image_id);
    setReviewSetId(reviewToOpen.annotation_set_id);
    setReviewBoxes(reviewToOpen.annotations.map(toReviewBox));
    setReviewHistory([]);
    setSelectedReviewBoxIndex(reviewToOpen.annotations.length > 0 ? 0 : null);
    setSelectedPoseKeypointIndex(0);
    setReviewToOpen(null);
  }, [reviewToOpen, selectedImageId]);

  const undoDraftBox = useCallback(() => {
    if (poseCollecting) {
      setPosePoints((current) => current.slice(0, -1));
      return;
    }
    if (pendingClassBox) {
      setPendingClassBox(null);
      return;
    }
    setDraftBoxes((current) => current.slice(0, -1));
  }, [pendingClassBox, poseCollecting]);

  const chooseLabelClass = (classId: number) => {
    setSelectedClassId(classId);
    if (poseProject || editableSet || reviewActionOnCurrentImage) return;
    if (pendingClassBox) {
      setDraftBoxes((current) => [...current, { ...pendingClassBox, class_id: classId }]);
      setPendingClassBox(null);
      setAnnotationSaveState("pending");
      return;
    }
    if (!drawingBox) setClassFilterId((current) => current === classId ? null : classId);
  };

  const rememberReviewBoxes = useCallback(() => {
    setReviewHistory((current) => [...current, {
      boxes: reviewBoxes.map(cloneReviewBox),
      selectedBoxIndex: selectedReviewBoxIndex,
      selectedPointIndex: selectedPoseKeypointIndex,
    }]);
  }, [reviewBoxes, selectedReviewBoxIndex, selectedPoseKeypointIndex]);

  const undoReviewEdit = useCallback(() => {
    setReviewHistory((current) => {
      const previous = current[current.length - 1];
      if (!previous) return current;
      setReviewBoxes(previous.boxes.map(cloneReviewBox));
      setSelectedReviewBoxIndex(previous.selectedBoxIndex);
      setSelectedPoseKeypointIndex(previous.selectedPointIndex);
      return current.slice(0, -1);
    });
  }, []);

  const deleteSelectedReviewBox = useCallback(() => {
    if (selectedReviewBoxIndex == null) return;
    rememberReviewBoxes();
    setReviewBoxes((current) => {
      const next = current.filter((_, index) => index !== selectedReviewBoxIndex);
      setSelectedReviewBoxIndex(next.length > 0 ? 0 : null);
      return next;
    });
    setSelectedPoseKeypointIndex(0);
  }, [rememberReviewBoxes, selectedReviewBoxIndex]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const undoActionKey = matchesShortcut(event, shortcuts, "undoAction");
      const undoFromEditorSelect = undoActionKey && !!editableSet && target?.tagName === "SELECT" &&
        !!target.closest(".review-editor");
      if (activeView !== "workspace" || workspaceSection !== "annotate" || busy ||
          document.querySelector('[role="dialog"][aria-modal="true"]') ||
          target?.isContentEditable ||
          (["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName ?? "") && !undoFromEditorSelect) ||
          event.repeat) return;
      if (undoActionKey) {
        event.preventDefault();
        if (editableSet && reviewHistory.length > 0) undoReviewEdit();
        else if (poseCollecting && posePoints.length > 0) undoDraftBox();
        else if (!editableSet && !poseCollecting && (pendingClassBox || draftBoxes.length > 0)) undoDraftBox();
        return;
      }
      const undoKey = matchesShortcut(event, shortcuts, "undoDraft") || matchesShortcut(event, shortcuts, "undoDraftAlt");
      const deleteKey = matchesShortcut(event, shortcuts, "deleteBox") || matchesShortcut(event, shortcuts, "deleteBoxAlt");
      if (editableSet ? !deleteKey : !undoKey) return;
      if (poseProject && poseCollecting && posePoints.length > 0) {
        event.preventDefault();
        undoDraftBox();
      } else if (editableSet && selectedReviewBoxIndex != null) {
        event.preventDefault();
        deleteSelectedReviewBox();
      } else if (!poseCollecting && (pendingClassBox || draftBoxes.length > 0)) {
        event.preventDefault();
        undoDraftBox();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeView, annotationSaveState, busy, deleteSelectedReviewBox, draftBoxes.length, pendingClassBox, editableSet,
    poseCollecting, posePoints.length, poseProject, reviewHistory.length, selectedReviewBoxIndex, shortcuts,
    undoDraftBox, undoReviewEdit, workspaceSection]);

  useEffect(() => {
    if (!autoLabelResults) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAutoLabelResults(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [autoLabelResults]);

  const updateOperationProgress = (detail: string, completed?: number, total?: number) => {
    setOperationProgress((current) => current && !current.error ? { ...current, detail, completed, total } : current);
  };

  const runAction = async (label: string, action: () => Promise<unknown>, options: { refresh?: boolean; showProgress?: boolean } = {}) => {
    setBusy(true);
    setStatus(`${label}...`);
    if (options.showProgress) setOperationProgress({ title: label, detail: "Starting…", startedAt: Date.now() });
    try {
      await action();
      if (options.refresh !== false) {
        if (options.showProgress) updateOperationProgress("Updating the workspace…");
        if (selectedProjectId) await refreshProject(selectedProjectId);
        if (selectedImageId) await refreshAnnotations(selectedImageId);
      }
      setStatus(`${label} done`);
      if (options.showProgress) setOperationProgress(null);
    } catch (error) {
      setStatus(errorMessage(error));
      if (options.showProgress) setOperationProgress((current) => current ? { ...current, error: errorMessage(error) } : null);
    } finally {
      setBusy(false);
    }
  };

  const activateProject = async (projectId: number) => {
    if ((draftBoxes.length > 0 || failedImageBoxes.length > 0) && !(await saveManualAnnotation())) return;
    await annotationSaveTail.current;
    await acceptQueueTail.current;
    if (failedBoxesByImage.current.size > 0 || [...acceptQueue.current.values()].some((item) => item.state === "failed")) {
      setStatus("Some annotations could not be saved or accepted. Retry or discard failed actions before switching projects.");
      return;
    }
    clearProjectState();
    setProjectActionsOpen(false);
    setSelectedProjectId(projectId);
    await refreshProject(projectId);
    setActiveView("workspace");
  };

  const createProject = (details: { name: string; description: string; taskType: "OBJECT_DETECTION" | "POSE_ESTIMATION" }) => runAction("Create project", async () => {
    const project = await api.createProject({
      name: details.name.trim(),
      description: details.description.trim() || undefined,
      task_type: details.taskType,
    });
    const nextProjects = await api.listProjects();
    setProjects(nextProjects);
    setNewProjectOpen(false);
    await activateProject(project.project_id);
  }, { refresh: false });

  const openProject = (projectId: number) => {
    void activateProject(projectId);
  };

  const openImageManager = () => {
    if (!selectedProjectId) return;
    void runAction("Load image manager", async () => {
      setManagedImages(await api.listImages(selectedProjectId, true));
      setSelectedManagedImageIds(new Set());
      setImageManagerFilter("all");
      setImageManagerSearch("");
      setImageManagerOpen(true);
    }, { refresh: false });
  };

  const updateManagedImagesStatus = (nextStatus: "UPLOADED" | "ARCHIVED") => {
    if (!selectedProjectId) return;
    const imageIds = managedImages
      .filter((image) => selectedManagedImageIds.has(image.image_id) && image.status !== nextStatus)
      .map((image) => image.image_id);
    if (imageIds.length === 0) return;
    const action = nextStatus === "ARCHIVED" ? "Archive" : "Restore";
    if (!window.confirm(`${action} ${imageIds.length} selected images?`)) return;
    void runAction(`${action} images`, async () => {
      await api.updateImagesStatus(selectedProjectId, imageIds, nextStatus);
      setManagedImages(await api.listImages(selectedProjectId, true));
      setSelectedManagedImageIds(new Set());
    });
  };

  const deleteManagedImagesPermanently = () => {
    if (!selectedProjectId) return;
    const imageIds = managedImages
      .filter((image) => selectedManagedImageIds.has(image.image_id) && image.status === "ARCHIVED")
      .map((image) => image.image_id);
    if (imageIds.length === 0) return;
    if (!window.confirm(
      `Permanently delete ${imageIds.length} archived images? This also removes their labels and cannot be undone.`,
    )) return;
    void runAction("Delete images permanently", async () => {
      await api.deleteImagesPermanently(selectedProjectId, imageIds);
      setManagedImages(await api.listImages(selectedProjectId, true));
      setSelectedManagedImageIds(new Set());
    });
  };

  const importData = (fileList: FileList | File[] | null, selectUploadedImage = false) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0 || !selectedProjectId) return;
    void runAction(`Import ${files.length} file${files.length === 1 ? "" : "s"}`, async () => {
      let lastUploadedImage: ImageItem | null = null;
      updateOperationProgress("Checking files…", 0, files.length);
      for (const [index, file] of files.entries()) {
        updateOperationProgress(`Processing ${file.name}…`, index, files.length);
        if (file.name.toLowerCase().endsWith(".zip")) {
          const inspection = await api.inspectArchive(file);
          if (inspection.archive_type === "YOLO") {
            setPendingImportArchive(file);
            setArchiveInspection(inspection);
            setImportLimit(String(inspection.labeled_image_count));
            setImportRemainingImages(true);
            setImportClassMode("all");
            setImportClassIndexes("");
            setImportClassNames(inspection.class_names.join(", "));
            updateOperationProgress(`Inspected ${file.name}`, index + 1, files.length);
            continue;
          }
          await api.importImageArchive(selectedProjectId, file);
          updateOperationProgress(`Imported ${file.name}`, index + 1, files.length);
          continue;
        }
        if (!file.type.startsWith("image/")) {
          throw new Error(`Unsupported file: ${file.name}`);
        }
        lastUploadedImage = await api.uploadImage(selectedProjectId, file);
        updateOperationProgress(`Imported ${file.name}`, index + 1, files.length);
      }
      if (selectUploadedImage && lastUploadedImage) await selectAnnotationImage(lastUploadedImage.image_id);
    }, { showProgress: true });
  };

  const confirmYoloImport = () => {
    const file = pendingImportArchive;
    if (!file || !selectedProjectId || !archiveInspection) return;
    const indexValue = importClassIndexes.trim();
    const importAllClasses = importClassMode === "all";
    if (!importAllClasses && !indexValue) {
      setStatus("Enter at least one YOLO class index");
      return;
    }
    const labelLimit = Number(importLimit);
    if (
      !Number.isInteger(labelLimit)
      || labelLimit < 0
      || labelLimit > archiveInspection.labeled_image_count
    ) {
      setStatus(`Label count must be between 0 and ${archiveInspection.labeled_image_count}`);
      return;
    }
    if (!importRemainingImages && labelLimit === 0) {
      setStatus("Choose at least one labeled image or import the remaining images without labels");
      return;
    }
    const selectedIndexes = importAllClasses ? undefined : indexValue;
    const selectedClassNames = importClassNames.trim() || undefined;
    void runAction("Import YOLO dataset", async () => {
      updateOperationProgress("Importing images and labels from the archive…");
      const result = await api.importYoloZip(
        selectedProjectId,
        file,
        selectedClassNames,
        labelLimit,
        selectedIndexes,
        importRemainingImages,
      );
      setStatus(
        `Imported ${result.imported_images} images, ${result.imported_annotation_sets} labels, ${result.imported_boxes} boxes`,
      );
      setPendingImportArchive(null);
      setArchiveInspection(null);
    }, { showProgress: true });
  };

  const removeDraftBox = (index: number) => {
    setDraftBoxes((current) => current.filter((_, itemIndex) => itemIndex !== index));
  };

  const startPoseCollection = () => {
    if (!selectedImageId || selectedClassId == null || editableSet || busy || poseCollecting || reviewActionOnCurrentImage) return;
    setPosePoints([]);
    setPoseVisibilityMode("visible");
    setPoseCollecting(true);
    setStatus("Mark each keypoint as visible, occluded, or outside the frame");
  };

  const addPosePoint = (point: { x: number; y: number }) => {
    if (selectedClassId == null || !poseCollecting) return;
    appendPosePoint({ ...point, visibility: poseVisibilityMode === "visible" ? 2 : 1 });
  };

  const appendPosePoint = (posePoint: PosePoint) => {
    if (selectedClassId == null || !poseCollecting) return;
    const nextPoints = [...posePoints, posePoint];
    if (nextPoints.length < 17) {
      setPosePoints(nextPoints);
      return;
    }
    const poseBox = poseBounds(nextPoints);
    if (poseBox) {
      setDraftBoxes((current) => [...current, {
        class_id: selectedClassId,
        ...poseBox,
        keypoints: nextPoints,
      }]);
      setStatus("Pose added. Save labels or add another pose.");
    } else {
      setStatus("A pose needs at least one visible or occluded keypoint");
    }
    setPosePoints([]);
    setPoseCollecting(false);
  };

  const skipPosePoint = () => {
    appendPosePoint({ x: 0, y: 0, visibility: 0 });
  };

  const addClass = () => {
    if (!selectedProjectId) return;
    const nextClassIndex = classes.reduce((maxIndex, labelClass) => Math.max(maxIndex, labelClass.class_index), -1) + 1;
    void runAction("Create class", async () => {
      const created = await api.createClass({
        project_id: selectedProjectId,
        name: className.trim() || `class_${classes.length}`,
        class_index: nextClassIndex,
        color: classColor,
      });
      setSelectedClassId(created.class_id);
      setClassName("");
      setClassColor(DEFAULT_CLASS_COLORS[(nextClassIndex + 1) % DEFAULT_CLASS_COLORS.length]);
    });
  };

  const startClassEdit = (labelClass: LabelClass) => {
    setSelectedClassId(labelClass.class_id);
    setEditingClassId(labelClass.class_id);
    setEditingClassName(labelClass.name);
    setEditingClassColor(labelClass.color);
  };

  const cancelClassEdit = () => {
    setEditingClassId(null);
    setEditingClassName("");
  };

  const saveClassEdit = () => {
    if (editingClassId == null || !editingClassName.trim()) return;
    void runAction("Update class", async () => {
      await api.updateClass(editingClassId, {
        name: editingClassName.trim(),
        color: editingClassColor,
      });
      cancelClassEdit();
    });
  };

  const deleteClass = (labelClass: LabelClass) => {
    if (!window.confirm(`Delete class "${labelClass.name}"?`)) return;
    void runAction("Delete class", () => api.deleteClass(labelClass.class_id));
  };

  const deleteSelectedProject = () => {
    if (!selectedProjectId) return;
    if (draftBoxes.length > 0 || queuedImageCount > 0 || failedImageCount > 0 || queuedAcceptCount > 0 || failedAcceptCount > 0) {
      setStatus("Save or retry all annotation actions before deleting this project");
      return;
    }
    const project = projects.find((item) => item.project_id === selectedProjectId);
    if (!window.confirm(`Permanently delete project "${project?.name ?? selectedProjectId}"? This removes its images, labels, datasets, WebP previews, trained/imported models, and saved training files. This cannot be undone.`)) return;
    void runAction("Delete project", async () => {
      await api.deleteProject(selectedProjectId);
      const nextProjects = await api.listProjects();
      setProjects(nextProjects);
      setSelectedProjectId(nextProjects[0]?.project_id ?? null);
      clearProjectState();
      setActiveView("dashboard");
    }, { refresh: false });
  };

  const saveManualAnnotation = async (): Promise<boolean> => {
    if (draftBoxes.length === 0 && failedImageBoxes.length === 0) return true;
    if (!selectedImageId || annotationImageId !== selectedImageId) {
      setAnnotationSaveState("error");
      setStatus("Cannot save labels until this image's annotations load");
      return false;
    }
    const imageId = selectedImageId;
    if (draftBoxes.length === 0 && queuedBoxesByImage.current.has(imageId)) return true;
    if (draftBoxes.length > 0 && lastQueuedDraft.current?.imageId === imageId && lastQueuedDraft.current.boxes === draftBoxes) return true;
    const projectId = selectedProjectId;
    const boxesToSave = draftBoxes;
    if (boxesToSave.length > 0) lastQueuedDraft.current = { imageId, boxes: boxesToSave };
    queuedBoxesByImage.current.set(imageId, [...(queuedBoxesByImage.current.get(imageId) ?? []), ...boxesToSave]);
    setSaveQueueRevision((current) => current + 1);
    setAnnotationSaveState("saving");
    setDraftBoxes((current) => current.slice(boxesToSave.length));
    const operation = annotationSaveTail.current.then(async () => {
      try {
        // Fetch at execution time: an earlier queued revision may have changed the approved set.
        const currentSets = await api.listAnnotations(imageId);
        const approved = [...currentSets].reverse().find((set) => set.status === "APPROVED") ?? null;
        const pendingIds = currentSets.filter((set) => set.source_type === "MODEL" && set.status === "PENDING")
          .map((set) => set.annotation_set_id);
        const retriedBoxes = failedBoxesByImage.current.get(imageId) ?? [];
        const newBoxes = [...retriedBoxes, ...boxesToSave].map((box) => ({ ...box, confidence: null }));
        // Each revision is a complete snapshot, not a patch to the previous labels.
        const saved = approved
          ? await api.editAnnotation(approved.annotation_set_id, {
            annotations: [
              ...approved.annotations.map((box) => ({ ...toReviewBox(box), confidence: null })),
              ...newBoxes,
            ],
          })
          : await api.createAnnotationSet(imageId, { annotations: newBoxes });
        let saveMessage = "Labels saved";
        // A human label replaces pending model suggestions, but failure to dismiss one must not duplicate saved boxes.
        if (pendingIds.length > 0) {
          try {
            await api.rejectAnnotationsBulk(pendingIds);
          } catch (error) {
            saveMessage = `Labels saved, but pending suggestions remain: ${errorMessage(error)}`;
          }
        }
        failedBoxesByImage.current.delete(imageId);
        if (selectedProjectIdRef.current === projectId) {
          setImageAnnotationStatuses((current) => [
            ...current.filter((item) => item.image_id !== imageId),
            {
              image_id: imageId,
              annotation_set_id: saved.annotation_set_id,
              source_type: approved ? "HUMAN_CORRECTED" : "HUMAN",
              status: "APPROVED",
              box_count: saved.annotations.length,
              is_negative: false,
              class_ids: [...new Set(saved.annotations.map((box) => box.class_id))],
            },
          ]);
          if (selectedImageIdRef.current === imageId) {
            void refreshAnnotations(imageId);
            setAnnotationSaveState((current) => current === "pending" ? current :
              (queuedBoxesByImage.current.get(imageId) ?? []).length > boxesToSave.length ? "saving" : "saved");
            setStatus(saveMessage);
          }
        }
        return true;
      } catch (error) {
        failedBoxesByImage.current.set(imageId, [...(failedBoxesByImage.current.get(imageId) ?? []), ...boxesToSave]);
        if (selectedImageIdRef.current === imageId) setAnnotationSaveState("error");
        setStatus(`Labels for image ${imageId} not saved: ${errorMessage(error)}`);
        return false;
      } finally {
        const remaining = (queuedBoxesByImage.current.get(imageId) ?? []).slice(boxesToSave.length);
        if (remaining.length > 0) queuedBoxesByImage.current.set(imageId, remaining);
        else queuedBoxesByImage.current.delete(imageId);
        setSaveQueueRevision((current) => current + 1);
      }
    });
    annotationSaveTail.current = operation.then(() => undefined, () => undefined);
    return operation;
  };

  useEffect(() => {
    if (poseProject || editableSet || drawingBox || annotationSaveState !== "pending" || draftBoxes.length === 0) return;
    const timer = window.setTimeout(() => { void saveManualAnnotation(); }, 900);
    return () => window.clearTimeout(timer);
  }, [annotationSaveState, draftBoxes, drawingBox, editableSet, poseProject, selectedImageId]);

  useEffect(() => {
    if (!poseProject && annotationSaveState === "saved" && draftBoxes.length > 0) setAnnotationSaveState("pending");
  }, [annotationSaveState, draftBoxes.length, poseProject]);

  useEffect(() => {
    if (draftBoxes.length === 0 && queuedImageCount === 0 && failedImageCount === 0 && queuedAcceptCount === 0 && failedAcceptCount === 0) return;
    const warnBeforeLeaving = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnBeforeLeaving);
    return () => window.removeEventListener("beforeunload", warnBeforeLeaving);
  }, [draftBoxes.length, queuedImageCount, failedImageCount, queuedAcceptCount, failedAcceptCount]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const poseKeypointSelectShortcut = poseProject && !!editableSet &&
        target?.dataset.poseKeypointSelect === "true" &&
        (matchesShortcut(event, shortcuts, "previousPoint") || matchesShortcut(event, shortcuts, "nextPoint") ||
          matchesShortcut(event, shortcuts, "poseEditVisible") || matchesShortcut(event, shortcuts, "poseEditOccluded") ||
          matchesShortcut(event, shortcuts, "poseEditOutside") ||
          matchesShortcut(event, shortcuts, "save"));
      if (activeView !== "workspace" || workspaceSection !== "annotate" || imageSelectorOpen || autoLabelResults ||
          document.querySelector('[role="dialog"][aria-modal="true"]') ||
          target?.isContentEditable ||
          (["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName ?? "") && !poseKeypointSelectShortcut)) return;
      if (busy) return;
      if (matchesShortcut(event, shortcuts, "save")) {
        event.preventDefault();
        if (event.repeat) return;
        if (editableSet) {
          saveEditedLabels();
          return;
        }
        if (poseProject && poseCollecting && posePoints.length > 0) {
          setStatus("Finish all 17 points or cancel this pose before saving");
          return;
        }
        void saveManualAnnotation();
        return;
      }
      if (matchesShortcut(event, shortcuts, "cancelEdit") && (editableSet || poseCollecting || pendingClassBox)) {
        event.preventDefault();
        if (event.repeat) return;
        if (editableSet) cancelReviewEditor();
        else if (pendingClassBox) setPendingClassBox(null);
        else if (posePoints.length === 0 || window.confirm("Cancel this pose and discard its unfinished points?")) {
          setPoseCollecting(false);
          setPosePoints([]);
        }
        return;
      }
      if (editableSet) {
        if (poseProject && selectedReviewBoxIndex != null) {
          if (matchesShortcut(event, shortcuts, "previousPoint") || matchesShortcut(event, shortcuts, "nextPoint")) {
            event.preventDefault();
            setSelectedPoseKeypointIndex((current) => Math.max(0, Math.min(16,
              current + (matchesShortcut(event, shortcuts, "previousPoint") ? -1 : 1))));
          } else if (matchesShortcut(event, shortcuts, "poseEditVisible")) {
            event.preventDefault();
            setPoseEditVisibilityMode("visible");
          } else if (matchesShortcut(event, shortcuts, "poseEditOccluded")) {
            event.preventDefault();
            setPoseEditVisibilityMode("occluded");
          } else if (matchesShortcut(event, shortcuts, "poseEditOutside")) {
            event.preventDefault();
            if (!event.repeat) updateSelectedPoseKeypoint(null, 0);
          }
        }
        return;
      }
      if (matchesShortcut(event, shortcuts, "editSavedLabels")) {
        event.preventDefault();
        if (event.repeat || !selectedImageId || annotationImageId !== selectedImageId || !latestApproved) return;
        if (poseCollecting || drawingBox || pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || annotationSaveState === "saving" || reviewActionOnCurrentImage) {
          setStatus("Finish or cancel the current label before editing saved labels");
        } else {
          startManualEditor();
        }
        return;
      }
      if (matchesShortcut(event, shortcuts, "verifyCurrentNegative")) {
        event.preventDefault();
        if (event.repeat || !selectedImageId || annotationImageId !== selectedImageId) return;
        if (poseCollecting || drawingBox || pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || annotationSaveState === "saving" || reviewActionOnCurrentImage) {
          setStatus("Finish or cancel the current label before verifying this image as negative");
        } else if (latestImageAnnotationSet?.is_negative) {
          setStatus("This image is already a verified negative");
        } else {
          confirmNegative(selectedImageId);
        }
        return;
      }
      const acceptKey = matchesShortcut(event, shortcuts, "acceptSuggestion");
      const rejectKey = matchesShortcut(event, shortcuts, "rejectSuggestion");
      const editKey = matchesShortcut(event, shortcuts, "editSuggestion");
      if (!poseCollecting && (acceptKey || rejectKey || editKey)) {
        if (pendingSuggestions.length === 0) return;
        event.preventDefault();
        if (event.repeat || reviewShortcutInFlight.current) return;
        if (pendingSuggestions.length !== 1) {
          setStatus("Multiple pending sets on this image; choose one in Review Queue");
          return;
        }
        if (pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || poseCollecting || annotationSaveState === "saving") {
          setStatus("Save or cancel the current draft before reviewing this suggestion");
          return;
        }
        const pendingSet = pendingSuggestions[0];
        if (reviewActionOnCurrentImage && !acceptKey) {
          setStatus("Resolve the queued or failed Accept before editing or rejecting this suggestion");
          return;
        }
        if (editKey) {
          startReviewEditor(pendingSet);
          return;
        }
        if (acceptKey) {
          void queueAccept(pendingSet);
          return;
        }
        reviewShortcutInFlight.current = true;
        void review(pendingSet, "reject").finally(() => {
          reviewShortcutInFlight.current = false;
        });
        return;
      }
      if (!poseCollecting && (matchesShortcut(event, shortcuts, "previousImage") || matchesShortcut(event, shortcuts, "nextImage"))) {
        event.preventDefault();
        navigateSelectedImage(matchesShortcut(event, shortcuts, "previousImage") ? -1 : 1);
      } else if (poseProject) {
        if (matchesShortcut(event, shortcuts, "addPose") && !poseCollecting && !event.repeat) {
          event.preventDefault();
          startPoseCollection();
        } else if (poseCollecting && matchesShortcut(event, shortcuts, "poseVisible")) {
          event.preventDefault();
          setPoseVisibilityMode("visible");
        } else if (poseCollecting && matchesShortcut(event, shortcuts, "poseOccluded")) {
          event.preventDefault();
          setPoseVisibilityMode("occluded");
        } else if (poseCollecting && matchesShortcut(event, shortcuts, "poseOutside") && !event.repeat) {
          event.preventDefault();
          skipPosePoint();
        }
      } else if (!poseProject) {
        const classIndex = Array.from({ length: 9 }, (_, index) => index)
          .find((index) => matchesShortcut(event, shortcuts, `class${index + 1}`));
        const labelClass = classIndex == null ? null : classes[classIndex];
        if (labelClass) {
          event.preventDefault();
          if (event.repeat) return;
          chooseLabelClass(labelClass.class_id);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeView, autoLabelResults, busy, classes, editableSet, imageSelectorOpen, poseProject, shortcuts, workspaceSection,
    selectedImageId, images, draftBoxes, pendingClassBox, annotationImageId, latestApproved, latestImageAnnotationSet, pendingSuggestions,
    poseCollecting, posePoints.length, drawingBox, selectedClassId, poseVisibilityMode, selectedReviewBoxIndex, selectedPoseKeypointIndex,
    annotationSaveState, failedImageBoxes.length, reviewActionOnCurrentImage, reviewSetId, selectedProjectId, reviewBoxes, editingManualLabels, reviewHistory.length]);

  const createAndBuildDataset = (imageIds: number[]) => {
    if (!selectedProjectId) return;
    void runAction("Create and build dataset version", async () => {
      updateOperationProgress("Creating dataset version…");
      const dataset = await api.createDataset({
        project_id: selectedProjectId,
        image_ids: imageIds,
        preprocess_config: { resize: { width: 640, height: 640 } },
        augmentation_config: { horizontal_flip: true },
        default_split: "TRAIN",
        auto_split: true,
      });
      updateOperationProgress("Building YOLO files from selected images…");
      await api.buildYolo(dataset.version_id);
      setSelectedDatasetVersionId(dataset.version_id);
      setDatasetSelectionOpen(false);
    }, { showProgress: true });
  };

  const deleteDatasetVersion = (dataset: DatasetVersion) => {
    if (!window.confirm(`Delete dataset version ${dataset.version_number}? This cannot be undone.`)) return;
    void runAction("Delete dataset version", async () => {
      await api.deleteDataset(dataset.version_id);
      setDatasetToView(null);
    });
  };

  const deleteModel = (model: ModelItem) => {
    setModelActionsOpenId(null);
    if (!window.confirm(`Delete Model v${model.version}? Its model files and exported formats will be permanently removed.`)) return;
    void runAction("Delete model", async () => {
      await api.deleteModel(model.model_id);
      setModelDownloadFormats((current) => {
        const next = { ...current };
        delete next[model.model_id];
        return next;
      });
    });
  };

  const importModel = () => {
    if (!selectedProjectId || !modelImportFile || busy) return;
    const file = modelImportFile;
    setModelImportResult(null);
    setModelImportError(null);
    void runAction("Import model", async () => {
      updateOperationProgress("Uploading and checking model weights and classes…");
      const result = await api.importModel(selectedProjectId, file);
      setModelImportResult(result);
      setModelImportFile(null);
      updateOperationProgress("Adding model and classes to this project…");
    }, { showProgress: true });
  };

  const trainYolo = () => {
    if (!selectedProjectId || !selectedDataset) return;
    setTrainingSetupOpen(false);
    void runAction("Train YOLO", async () => {
      updateOperationProgress("Creating training run…");
      const run = await api.createTrainingRun({
        project_id: selectedProjectId,
        version_id: selectedDataset.version_id,
        model_type: selectedProjectTask === "POSE_ESTIMATION" ? "YOLO26_POSE" : "YOLO26_DETECT",
        base_model: trainingSettings.baseModel,
        training_config: {
          epochs: trainingSettings.epochs,
          imgsz: trainingSettings.imgsz,
          batch: trainingSettings.batch,
          patience: trainingSettings.patience,
          optimizer: trainingSettings.optimizer,
          lr0: trainingSettings.lr0,
          workers: trainingSettings.workers,
          device: trainingSettings.device,
          cache: trainingSettings.cache,
          amp: trainingSettings.amp,
          cos_lr: trainingSettings.cos_lr,
          val: true,
          plots: true,
          verbose: true,
        },
      });
      updateOperationProgress("Preparing data and training the model. This can take a while…");
      await api.startYoloTraining(run.training_run_id);
    }, { showProgress: true });
  };

  const resumeYolo = (runId: number) => {
    void runAction("Resume training", async () => {
      updateOperationProgress("Loading checkpoint and resuming training…");
      await api.resumeYoloTraining(runId);
    }, { showProgress: true });
  };

  const chooseTestImages = (fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0) return;
    const knownFingerprints = new Set(testImages.map((image) => image.fingerprint));
    const added: TestImage[] = [];
    let duplicateCount = 0;
    files.forEach((file) => {
      const fingerprint = `${file.name}-${file.size}-${file.lastModified}`;
      if (knownFingerprints.has(fingerprint)) {
        duplicateCount += 1;
        return;
      }
      knownFingerprints.add(fingerprint);
      added.push({
        key: `${fingerprint}-${crypto.randomUUID()}`,
        fingerprint,
        file,
        url: URL.createObjectURL(file),
      });
    });
    if (added.length === 0) {
      setStatus(`Skipped ${duplicateCount} duplicate image${duplicateCount === 1 ? "" : "s"}`);
      return;
    }
    setTestImages((current) => [...current, ...added]);
    setTestDatasetSource((current) => current && !current.includes(" + uploaded images") ? `${current} + uploaded images` : current);
    setTestPredictions((current) => [
      ...Array.from({ length: testImages.length }, (_, index) => current[index] ?? null),
      ...Array<ModelPrediction | null>(added.length).fill(null),
    ]);
    setTestProgress((current) => ({
      ...current,
      total: testImages.length + added.length,
      running: false,
      currentFrom: 0,
      currentTo: 0,
      currentImageIndex: null,
    }));
    setStatus(`Added ${added.length} test image${added.length === 1 ? "" : "s"}${duplicateCount > 0 ? ` · skipped ${duplicateCount} duplicate${duplicateCount === 1 ? "" : "s"}` : ""}`);
  };

  const clearTestImages = () => {
    testImages.forEach((image) => URL.revokeObjectURL(image.url));
    setTestImages([]);
    setTestDatasetSource(null);
    setTestPredictions([]);
    setTestFailures({});
    setTestProgress({ processed: 0, total: 0, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
    setStatus("Cleared test images");
  };

  const loadDatasetTestImages = (versionId: number) => {
    const dataset = datasets.find((version) => version.version_id === versionId && version.project_id === selectedProjectId);
    const testItems = dataset?.items.filter((item) => item.split === "TEST") ?? [];
    if (!dataset || testItems.length === 0) {
      setStatus("This dataset version has no test images");
      return;
    }
    if (testImages.length > 0 && !window.confirm(`Replace the ${testImages.length} current test images and results with ${testItems.length} images from Dataset v${dataset.version_number}?`)) return;
    void runAction("Load dataset test images", async () => {
      const projectId = selectedProjectId;
      if (projectId == null) throw new Error("Select a project first");
      const requestId = projectRequestId.current;
      const projectImages = await api.listImages(projectId, true);
      if (requestId !== projectRequestId.current) throw new Error("Project changed while loading test images");
      const imageById = new Map(projectImages.map((image) => [image.image_id, image]));
      const missingIds = testItems.filter((item) => !imageById.has(item.image_id)).map((item) => item.image_id);
      if (missingIds.length > 0) throw new Error(`${missingIds.length} dataset test images are unavailable in this project. Refresh the project and try again.`);
      const loaded: TestImage[] = [];
      try {
        for (let offset = 0; offset < testItems.length; offset += 6) {
          const batch = testItems.slice(offset, offset + 6);
          const blobs = await Promise.all(batch.map((item) => api.getImageBlob(imageById.get(item.image_id)!)));
          if (requestId !== projectRequestId.current) throw new Error("Project changed while loading test images");
          batch.forEach((item, index) => {
            const image = imageById.get(item.image_id)!;
            const file = new File([blobs[index]], image.original_filename || `image-${item.image_id}.jpg`, { type: blobs[index].type || "image/jpeg" });
            const fingerprint = `dataset:${dataset.version_id}:${item.image_id}`;
            loaded.push({ key: fingerprint, fingerprint, file, url: URL.createObjectURL(file) });
          });
          updateOperationProgress(`Loading ${loaded.length} of ${testItems.length} test images…`, loaded.length, testItems.length);
        }
        testImages.forEach((image) => URL.revokeObjectURL(image.url));
        setTestImages(loaded);
        setTestDatasetSource(`Dataset v${dataset.version_number} · Test split`);
        setTestPredictions([]);
        setTestFailures({});
        setTestProgress({ processed: 0, total: loaded.length, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
      } catch (error) {
        loaded.forEach((image) => URL.revokeObjectURL(image.url));
        throw error;
      }
    }, { refresh: false, showProgress: true });
  };

  const removeTestImage = (imageIndex: number) => {
    URL.revokeObjectURL(testImages[imageIndex].url);
    const nextImages = testImages.filter((_, index) => index !== imageIndex);
    const nextPredictions = testPredictions.filter((_, index) => index !== imageIndex);
    const nextFailures = Object.fromEntries(
      Object.entries(testFailures).flatMap(([indexText, message]) => {
        const index = Number(indexText);
        if (index === imageIndex) return [];
        return [[index > imageIndex ? index - 1 : index, message]];
      }),
    );
    setTestImages(nextImages);
    if (nextImages.length === 0) setTestDatasetSource(null);
    setTestPredictions(nextPredictions);
    setTestFailures(nextFailures);
    setTestProgress({
      processed: nextPredictions.filter(Boolean).length + Object.keys(nextFailures).length,
      total: nextImages.length,
      running: false,
      elapsedMs: 0,
      currentFrom: 0,
      currentTo: 0,
      currentImageIndex: null,
    });
  };

  const changeTestModel = (modelId: number | "pose-pretrained" | null) => {
    setSelectedTestModelId(modelId);
    setTestPredictions([]);
    setTestFailures({});
    setTestProgress({ processed: 0, total: testImages.length, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
  };

  const changeTestThreshold = (value: number) => {
    setTestConfidenceThreshold(value);
    setTestPredictions([]);
    setTestFailures({});
    setTestProgress({ processed: 0, total: testImages.length, running: false, elapsedMs: 0, currentFrom: 0, currentTo: 0, currentImageIndex: null });
  };

  const runModelTest = (mode: "untested" | "all" = "all") => {
    if (!selectedTestModelId || testImages.length === 0 || !selectedProjectId) return;
    const targetIndexes = testImages.flatMap((_, index) => (
      mode === "all" || (testPredictions[index] == null && testFailures[index] == null) ? [index] : []
    ));
    if (targetIndexes.length === 0) {
      setStatus("All selected images have already been tested");
      return;
    }
    void runAction("Run model test", async () => {
      const batchSize = 100;
      const results: (ModelPrediction | null)[] = mode === "all"
        ? Array(testImages.length).fill(null)
        : Array.from({ length: testImages.length }, (_, index) => testPredictions[index] ?? null);
      const failures: Record<number, string> = mode === "all" ? {} : { ...testFailures };
      targetIndexes.forEach((index) => { delete failures[index]; });
      const startedAt = performance.now();
      let processed = mode === "all"
        ? 0
        : results.filter(Boolean).length + Object.keys(failures).length;
      const initialProcessed = processed;
      let currentFrom = targetIndexes[0] + 1;
      let currentTo = currentFrom;
      let currentImageIndex: number | null = targetIndexes[0];
      const resultDisplayDelay = targetIndexes.length <= 30 ? 45 : targetIndexes.length <= 100 ? 12 : 0;
      setTestPredictions([...results]);
      setTestFailures({ ...failures });
      setTestProgress({ processed, total: testImages.length, running: true, elapsedMs: 0, currentFrom, currentTo, currentImageIndex });

      const predictFiles = (files: File[], onResult: (index: number, result: ModelPrediction) => void | Promise<void>) => selectedTestModelId === "pose-pretrained"
        ? api.streamPretrainedPosePredictions(selectedProjectId, files, testConfidenceThreshold, (event) => onResult(event.index, event.result))
        : api.streamModelPredictions(selectedTestModelId, files, testConfidenceThreshold, (event) => onResult(event.index, event.result));
      const publishProgress = () => {
        setTestPredictions([...results]);
        setTestFailures({ ...failures });
        updateOperationProgress(
          processed === initialProcessed ? "Loading model and preparing test images…" : `Processing image ${currentFrom} of ${testImages.length}…`,
          processed - initialProcessed,
          targetIndexes.length,
        );
        setTestProgress({
          processed,
          total: testImages.length,
          running: processed < testImages.length,
          elapsedMs: performance.now() - startedAt,
          currentFrom,
          currentTo,
          currentImageIndex,
        });
      };

      try {
        for (let offset = 0; offset < targetIndexes.length; offset += batchSize) {
          const batchIndexes = targetIndexes.slice(offset, offset + batchSize);
          const batch = batchIndexes.map((index) => testImages[index]);
          const files = batch.map((image) => image.file);
          currentFrom = batchIndexes[0] + 1;
          currentTo = currentFrom;
          currentImageIndex = batchIndexes[0];
          publishProgress();
          setStatus(`Testing ${offset + 1}-${Math.min(offset + batch.length, targetIndexes.length)} of ${targetIndexes.length}`);
          try {
            await predictFiles(files, async (index, result) => {
              const resultIndex = batchIndexes[index];
              if (results[resultIndex] == null) processed += 1;
              results[resultIndex] = result;
              const nextTarget = targetIndexes[offset + index + 1];
              currentFrom = nextTarget == null ? resultIndex + 1 : nextTarget + 1;
              currentTo = currentFrom;
              currentImageIndex = nextTarget ?? null;
              publishProgress();
              if (resultDisplayDelay > 0) await sleep(resultDisplayDelay);
            });
          } catch {
            for (let index = 0; index < batch.length; index += 1) {
              const resultIndex = batchIndexes[index];
              if (results[resultIndex] != null) continue;
              currentFrom = resultIndex + 1;
              currentTo = resultIndex + 1;
              currentImageIndex = resultIndex;
              publishProgress();
              try {
                await predictFiles([batch[index].file], (_streamIndex, result) => {
                  results[resultIndex] = result;
                });
                if (!results[resultIndex]) failures[resultIndex] = "No result returned";
              } catch (error) {
                failures[resultIndex] = errorMessage(error);
              }
              processed += 1;
              publishProgress();
            }
          }
        }
      } finally {
        setTestProgress({
          processed,
          total: testImages.length,
          running: false,
          elapsedMs: performance.now() - startedAt,
          currentFrom: 0,
          currentTo: 0,
          currentImageIndex: null,
        });
      }
    }, { refresh: false, showProgress: true });
  };

  const showAutoLabelResults = async (job: AutoLabelJob) => {
    const sourceModel = models.find((model) => model.model_id === job.model_id);
    setAutoLabelResultsModelName(job.base_model
      ? job.base_model === "yolo26x-pose.pt" ? "YOLO26 Pose Extra Large pretrained" : "YOLO26 Pose Nano pretrained"
      : sourceModel ? `Model v${sourceModel.version}` : `Model #${job.model_id}`);
    const resultItems = await Promise.all(job.items.map(async (item) => {
      const image = images.find((candidate) => candidate.image_id === item.image_id);
      if (!image) return null;
      let imageAnnotations: AnnotationSet[] = [];
      if (item.annotation_set_id != null) {
        try {
          imageAnnotations = await api.listAnnotations(item.image_id);
        } catch {
          imageAnnotations = [];
        }
      }
      const annotationSet = resolveEffectiveAnnotationSet(imageAnnotations, item.annotation_set_id);
      return {
        image,
        annotationSet,
        status: item.status,
      } satisfies AutoLabelResult;
    }));
    setAutoLabelResults(resultItems.filter((item): item is AutoLabelResult => item != null));
  };

  const autoLabel = (confidenceThreshold: number) => {
    const selectedModel = typeof selectedAutoLabelModelId === "number"
      ? models.find((model) => model.model_id === selectedAutoLabelModelId)
      : null;
    if (!selectedProjectId || !selectedAutoLabelModelId || images.length === 0 ||
        !Number.isFinite(confidenceThreshold) || confidenceThreshold < 0.01 || confidenceThreshold > 1) return;
    setAutoLabelSettingsOpen(false);
    void runAction("Auto label", async () => {
      updateOperationProgress("Creating Auto label job…");
      const job = await api.createAutoLabelJob({
        project_id: selectedProjectId,
        model_id: selectedModel?.model_id ?? null,
        base_model: selectedAutoLabelModelId === "pose-pretrained" ? "yolo26x-pose.pt" : null,
        image_ids: images.map((image) => image.image_id),
        confidence_threshold: confidenceThreshold,
        max_predictions_per_image: 3,
        only_unlabeled: true,
      });
      updateOperationProgress("Starting the model…", 0, job.total_images ?? undefined);
      await api.startYoloAutoLabelAsync(job.job_id);
      await waitForAutoLabelJob(job.job_id);
    }, { showProgress: true });
  };

  const waitForAutoLabelJob = async (jobId: number) => {
    for (;;) {
      await sleep(1000);
      const job = await api.getAutoLabelJob(jobId);
      setStatus(`Auto label ${job.processed_images ?? 0}/${job.total_images ?? 0} ${job.status}`);
      updateOperationProgress(
        job.processed_images ? "Generating suggestions…" : "Loading model and preparing images…",
        job.processed_images ?? 0,
        job.total_images ?? undefined,
      );
      if (selectedProjectId) setAutoJobs(await api.listAutoLabelJobs(selectedProjectId));
      if (job.status === "COMPLETED") {
        updateOperationProgress("Loading Auto label results…");
        await showAutoLabelResults(job);
        return;
      }
      if (job.status === "FAILED") {
        throw new Error(job.error_message ?? "Auto label failed");
      }
    }
  };

  const closeReviewEditor = () => {
    setReviewSetId(null);
    setManualEditSetId(null);
    setReviewBoxes([]);
    setReviewHistory([]);
    setSelectedReviewBoxIndex(null);
    setSelectedPoseKeypointIndex(0);
    setPoseEditVisibilityMode("visible");
  };

  const cancelReviewEditor = () => {
    if (reviewHistory.length > 0 && !window.confirm("Discard unsaved label changes?")) return;
    closeReviewEditor();
  };

  const startReviewEditor = (set: AnnotationSet) => {
    setDraftBoxes([]);
    setPendingClassBox(null);
    setManualEditSetId(null);
    setReviewSetId(set.annotation_set_id);
    setReviewBoxes(set.annotations.map(toReviewBox));
    setReviewHistory([]);
    setSelectedReviewBoxIndex(set.annotations.length > 0 ? 0 : null);
    setSelectedPoseKeypointIndex(0);
    setPoseEditVisibilityMode("visible");
  };

  const startManualEditor = () => {
    if (!latestApproved) return;
    setDraftBoxes([]);
    setPendingClassBox(null);
    setReviewSetId(null);
    setManualEditSetId(latestApproved.annotation_set_id);
    setReviewBoxes(latestApproved.annotations.map(toReviewBox));
    setReviewHistory([]);
    setSelectedReviewBoxIndex(latestApproved.annotations.length > 0 ? 0 : null);
    setSelectedPoseKeypointIndex(0);
    setPoseEditVisibilityMode("visible");
  };

  const openAutoLabelResult = (result: AutoLabelResult) => {
    void selectAnnotationImage(result.image.image_id).then((selected) => {
      if (!selected) return;
      setAutoLabelResults(null);
      if (result.annotationSet) setReviewToOpen(result.annotationSet);
    });
  };

  const rejectAllAutoLabelResults = () => {
    if (!autoLabelResults) return;
    const pendingIds = autoLabelResults.flatMap((result) => (
      result.annotationSet?.status === "PENDING" ? [result.annotationSet.annotation_set_id] : []
    ));
    if (pendingIds.length === 0) return;
    if (!window.confirm(`Reject all ${pendingIds.length} pending auto-label results?`)) return;
    void runAction(`Reject ${pendingIds.length} auto-label results`, async () => {
      await api.rejectAnnotationsBulk(pendingIds);
      setAutoLabelResults(null);
      closeReviewEditor();
    });
  };

  const acceptAllAutoLabelResults = () => {
    if (!autoLabelResults) return;
    const pendingIds = autoLabelResults.flatMap((result) => (
      result.annotationSet?.status === "PENDING" ? [result.annotationSet.annotation_set_id] : []
    ));
    if (pendingIds.length === 0) return;
    if (!window.confirm(`Accept all ${pendingIds.length} pending auto-label results? Images with no boxes will stay out of the dataset until you verify them as negative.`)) return;
    const pendingIdSet = new Set(pendingIds);
    void runAction(`Accept ${pendingIds.length} auto-label results`, async () => {
      await api.acceptAnnotationsBulk(pendingIds);
      setAutoLabelResults((current) => current?.map((result) => (
        result.annotationSet && pendingIdSet.has(result.annotationSet.annotation_set_id)
          ? { ...result, annotationSet: { ...result.annotationSet, status: "APPROVED" } }
          : result
      )) ?? null);
      closeReviewEditor();
    });
  };

  const confirmNegative = (imageId: number) => {
    void runAction("Confirm negative image", async () => {
      const verified = await api.confirmNegative(imageId);
      if (selectedImageId === imageId && annotationImageId === imageId) {
        setAnnotations((current) => [
          ...current.filter((set) => set.annotation_set_id !== verified.annotation_set_id).map((set) =>
            set.source_type === "MODEL" && set.status === "PENDING" ? { ...set, status: "REJECTED" } : set
          ),
          verified,
        ]);
      }
      setImageAnnotationStatuses((current) => [
        ...current.filter((status) => status.image_id !== imageId),
        {
          image_id: imageId,
          annotation_set_id: verified.annotation_set_id,
          source_type: "HUMAN",
          status: "APPROVED",
          box_count: 0,
          is_negative: true,
          class_ids: [],
        },
      ]);
    }, { refresh: false });
  };

  const reviewAllProjectSuggestions = (mode: "accept" | "reject") => {
    const pendingIds = pendingProjectAnnotationSetIds;
    if (pendingIds.length === 0) return;
    const action = mode === "accept" ? "Accept" : "Reject";
    if (!window.confirm(`${action} all ${pendingIds.length} pending auto-label results?`)) return;
    void runAction(`${action} ${pendingIds.length} auto-label results`, async () => {
      if (mode === "accept") await api.acceptAnnotationsBulk(pendingIds);
      else await api.rejectAnnotationsBulk(pendingIds);
      closeReviewEditor();
    });
  };

  const reviewSelectedProjectSuggestions = (imageIds: number[], mode: "accept" | "reject") => {
    const selectedImageIds = new Set(imageIds);
    const selectedPendingStatuses = pendingImageAnnotationStatuses.filter(
      (status) => selectedImageIds.has(status.image_id),
    );
    const pendingIds = selectedPendingStatuses.map((status) => status.annotation_set_id);
    if (pendingIds.length === 0) return;
    const action = mode === "accept" ? "Accept" : "Reject";
    if (!window.confirm(`${action} auto-label results for ${selectedPendingStatuses.length} selected images?`)) return;
    void runAction(`${action} ${selectedPendingStatuses.length} selected auto-label results`, async () => {
      if (mode === "accept") await api.acceptAnnotationsBulk(pendingIds);
      else await api.rejectAnnotationsBulk(pendingIds);
      closeReviewEditor();
    });
  };

  const resetReviewBoxes = () => {
    if (!editableSet) return;
    rememberReviewBoxes();
    setReviewBoxes(editableSet.annotations.map(toReviewBox));
    setSelectedReviewBoxIndex(editableSet.annotations.length > 0 ? 0 : null);
    setSelectedPoseKeypointIndex(0);
  };

  const addReviewBox = (box: DraftBox) => {
    rememberReviewBoxes();
    setReviewBoxes((current) => [...current, { ...box, confidence: null }]);
    setSelectedReviewBoxIndex(reviewBoxes.length);
  };

  const updateReviewBox = (index: number, box: ReviewBox) => {
    setReviewBoxes((current) => current.map((item, itemIndex) => itemIndex === index ? box : item));
  };

  const changeSelectedReviewClass = (classId: number) => {
    if (selectedReviewBoxIndex == null) return;
    rememberReviewBoxes();
    setReviewBoxes((current) => current.map((box, index) => (
      index === selectedReviewBoxIndex ? { ...box, class_id: classId, confidence: null } : box
    )));
  };

  const updateSelectedPoseKeypoint = (point: { x: number; y: number } | null, visibility: number) => {
    if (selectedReviewBoxIndex == null) return;
    rememberReviewBoxes();
    setReviewBoxes((current) => current.map((box, boxIndex) => {
      if (boxIndex !== selectedReviewBoxIndex) return box;
      const keypoints = normalizePosePoints(box.keypoints);
      keypoints[selectedPoseKeypointIndex] = visibility === 0
        ? { x: 0, y: 0, visibility: 0 }
        : { x: point?.x ?? 0, y: point?.y ?? 0, visibility };
      const bounds = poseBounds(keypoints);
      return {
        ...box,
        ...(bounds ?? {}),
        confidence: null,
        keypoints,
      };
    }));
    setSelectedPoseKeypointIndex((current) => Math.min(16, current + 1));
  };

  const saveEditedLabels = () => {
    if (!editableSet) return;
    const annotationSetId = editableSet.annotation_set_id;
    void runAction(editingManualLabels ? "Save edited labels" : "Save corrected labels", async () => {
      const payload = {
        annotations: reviewBoxes.map((box) => ({ ...box, confidence: null })),
      };
      if (editingManualLabels) await api.editAnnotation(annotationSetId, payload);
      else await api.correctAnnotation(annotationSetId, payload);
      closeReviewEditor();
    });
  };

  const queueAccept = (set: AnnotationSet): Promise<void> => {
    if (selectedProjectId == null || acceptQueue.current.get(set.annotation_set_id)?.state === "queued") return Promise.resolve();
    const projectId = selectedProjectId;
    acceptQueue.current.set(set.annotation_set_id, { set, projectId, state: "queued" });
    setAcceptQueueRevision((current) => current + 1);
    setStatus(`Accept queued for image ${set.image_id}; you can move to another image`);
    const operation = acceptQueueTail.current.then(async () => {
      try {
        const accepted = await api.acceptAnnotation(set.annotation_set_id);
        acceptQueue.current.delete(set.annotation_set_id);
        if (selectedProjectIdRef.current === projectId) {
          setImageAnnotationStatuses((current) => [
            ...current.filter((item) => item.image_id !== set.image_id),
            {
              image_id: set.image_id,
              annotation_set_id: accepted.annotation_set_id,
              source_type: "MODEL",
              status: "APPROVED",
              box_count: accepted.annotations.length,
              is_negative: accepted.is_negative,
              class_ids: accepted.is_negative ? [] : [...new Set([
                ...(current.find((item) => item.image_id === set.image_id)?.class_ids ?? []),
                ...accepted.annotations.map((box) => box.class_id),
              ])],
            },
          ]);
          if (selectedImageIdRef.current === set.image_id) {
            setAnnotations((current) => current.map((item) => item.annotation_set_id === accepted.annotation_set_id ? accepted : item));
            void refreshAnnotations(set.image_id);
          }
          setStatus(`Accepted image ${set.image_id}`);
        }
      } catch (error) {
        acceptQueue.current.set(set.annotation_set_id, { set, projectId, state: "failed", error: errorMessage(error) });
        setStatus(`Accept failed for image ${set.image_id}: ${errorMessage(error)}`);
      } finally {
        setAcceptQueueRevision((current) => current + 1);
      }
    });
    acceptQueueTail.current = operation.then(() => undefined, () => undefined);
    return operation;
  };

  const discardFailedAccept = (setId: number) => {
    const failed = acceptQueue.current.get(setId);
    if (failed?.state !== "failed") return;
    void api.listAnnotations(failed.set.image_id).then((sets) => {
      if (selectedProjectIdRef.current !== failed.projectId) return;
      acceptQueue.current.delete(setId);
      setAcceptQueueRevision((current) => current + 1);
      const latest = sets[sets.length - 1];
      const latestApprovedSet = [...sets].reverse().find((item) => item.status === "APPROVED");
      const activeSets = [...(latestApprovedSet ? [latestApprovedSet] : []), ...sets.filter((item) => item.status === "PENDING")];
      setImageAnnotationStatuses((current) => [
        ...current.filter((item) => item.image_id !== failed.set.image_id),
        ...(latest ? [{
          image_id: failed.set.image_id,
          annotation_set_id: latest.annotation_set_id,
          source_type: latest.source_type as ImageAnnotationStatus["source_type"],
          status: latest.status as ImageAnnotationStatus["status"],
          box_count: latest.annotations.length,
          is_negative: latest.is_negative,
          class_ids: [...new Set(activeSets.flatMap((item) => item.annotations.map((box) => box.class_id)))].sort((a, b) => a - b),
        }] : []),
      ]);
      if (selectedImageIdRef.current === failed.set.image_id) {
        setAnnotations(sets);
        setAnnotationImageId(failed.set.image_id);
      }
      setStatus("Failed accept discarded; image status refreshed from server");
    }).catch((error) => setStatus(`Could not refresh image status: ${errorMessage(error)}`));
  };

  const review = (set: AnnotationSet, mode: "accept" | "reject") => {
    if (mode === "accept") return queueAccept(set);
    return runAction(`${mode} suggestion`, async () => {
      await api.rejectAnnotation(set.annotation_set_id);
      if (set.annotation_set_id === reviewSetId) closeReviewEditor();
    });
  };

  const logout = async () => {
    if ((draftBoxes.length > 0 || failedImageBoxes.length > 0) && !(await saveManualAnnotation())) return;
    await annotationSaveTail.current;
    await acceptQueueTail.current;
    if (failedBoxesByImage.current.size > 0 || [...acceptQueue.current.values()].some((item) => item.state === "failed")) {
      setStatus("Some annotation actions failed. Retry or discard them before logging out.");
      return;
    }
    const refreshToken = session?.refresh_token;
    clearAuthSession();
    setSession(null);
    clearProjectState();
    setProjects([]);
    if (refreshToken) void api.logout(refreshToken).catch(() => undefined);
  };

  const download = async (label: string, getBlob: () => Promise<Blob>, filename: string) => {
    await runAction(label, async () => {
      updateOperationProgress("Preparing download…");
      const url = URL.createObjectURL(await getBlob());
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }, { refresh: false, showProgress: true });
  };

  if (!authChecked) return <div className="auth-loading">Checking your session...</div>;
  if (!session) {
    return <AuthScreen onAuthenticated={(nextSession) => {
      saveAuthSession(nextSession);
      setSession(nextSession);
      setAuthChecked(false);
    }} />;
  }

  return (
    <div className="app-shell">
      <header className="topbar global-header">
        <button className="brand-home platform-brand" onClick={() => { setProjectActionsOpen(false); setActiveView("dashboard"); }} disabled={busy} title="Back to project dashboard">
          <span className="brand-mark"><CircleDot size={19} /></span>
          <span className="brand-copy"><strong>CV Platform</strong><small>{status}</small></span>
        </button>
        <div className="topbar-actions">
          <button type="button" className="shortcut-settings-trigger" onClick={() => setShortcutsOpen(true)} disabled={busy} title="View and customize keyboard shortcuts"><Keyboard size={17} /> <span>Shortcuts</span></button>
          <span className="account-name user-chip">{session.user.username || session.user.email}</span>
          <button onClick={logout} disabled={busy}>Sign out</button>
          <button className="icon-button" onClick={() => void refreshAll()} disabled={busy} title="Refresh"><RefreshCw size={18} /></button>
        </div>
      </header>

      {activeView === "dashboard" ? (
        <ProjectDashboard
          projects={projects}
          busy={busy}
          onOpenCreate={() => setNewProjectOpen(true)}
          onOpen={openProject}
        />
      ) : <div className="workspace-page">
        <section className="project-context">
          <div className="project-context-main">
            <button className="back-button" onClick={() => { setProjectActionsOpen(false); setActiveView("dashboard"); }} disabled={busy} type="button">
              <ArrowLeft size={17} /> Projects
            </button>
            <div className="project-heading">
              <span>Projects / {selectedProject?.name ?? "Project"}</span>
              <div>
                <h1>{selectedProject?.name ?? "Project"}</h1>
                <span className="task-pill">{selectedProjectTask.replace(/_/g, " ")}</span>
              </div>
            </div>
          </div>
          <div className="project-context-actions" ref={projectActionsRef}>
            <button className="icon-button" onClick={() => setProjectActionsOpen((current) => !current)} aria-expanded={projectActionsOpen} title="Project actions" type="button">
              <MoreHorizontal size={19} />
            </button>
            {projectActionsOpen && (
              <div className="project-actions-menu">
                <button className="danger-item" onClick={() => { setProjectActionsOpen(false); void deleteSelectedProject(); }} disabled={!selectedProjectId || busy} type="button">
                  <Trash2 size={15} /> Delete project
                </button>
              </div>
            )}
          </div>
        </section>

        <div className="workspace-layout">
          <aside className="workspace-sidebar">
            <nav aria-label="Project sections">
              <button className={workspaceSection === "overview" ? "active" : ""} onClick={() => setWorkspaceSection("overview")} type="button"><BarChart3 size={17} /> Overview</button>
              <button className={workspaceSection === "data" ? "active" : ""} onClick={() => setWorkspaceSection("data")} type="button"><Database size={17} /> Data</button>
              <button className={workspaceSection === "annotate" ? "active" : ""} onClick={() => setWorkspaceSection("annotate")} type="button"><Boxes size={17} /> Annotate</button>
              <div className={`workspace-nav-group ${["train", "import-model", "models", "test"].includes(workspaceSection) ? "active" : ""}`}>
                <span><Layers3 size={17} /> Models</span>
                <button className={workspaceSection === "train" ? "active" : ""} onClick={() => setWorkspaceSection("train")} type="button"><Settings2 size={15} /> Train</button>
                <button className={workspaceSection === "import-model" ? "active" : ""} onClick={() => setWorkspaceSection("import-model")} type="button"><Upload size={15} /> Import model</button>
                <button className={workspaceSection === "models" ? "active" : ""} onClick={() => setWorkspaceSection("models")} type="button"><Download size={15} /> Download</button>
                <button className={workspaceSection === "test" ? "active" : ""} onClick={() => setWorkspaceSection("test")} type="button"><TestTube2 size={15} /> Test</button>
              </div>
            </nav>
          </aside>

          <div className="workspace-content">
          {workspaceSection === "overview" && (
            <main className="section-page">
              <PageHeading eyebrow="PROJECT OVERVIEW" title="Project activity" description="Your data, annotation, and model workflow at a glance." />
              <section className="overview-stat-grid">
                <button className="overview-stat" onClick={() => setWorkspaceSection("data")} type="button"><ImageIcon size={20} /><span>Images</span><strong>{images.length}</strong><small>Manage source data <ChevronRight size={13} /></small></button>
                <button className="overview-stat" onClick={() => setWorkspaceSection("annotate")} type="button"><Boxes size={20} /><span>Classes</span><strong>{classes.length}</strong><small>Open annotation <ChevronRight size={13} /></small></button>
                <button className="overview-stat" onClick={() => setWorkspaceSection("train")} type="button"><Database size={20} /><span>Datasets</span><strong>{datasets.length}</strong><small>Configure training <ChevronRight size={13} /></small></button>
                <button className="overview-stat" onClick={() => setWorkspaceSection("models")} type="button"><Layers3 size={20} /><span>Models</span><strong>{models.length}</strong><small>Review models <ChevronRight size={13} /></small></button>
              </section>
              <section className="section-panel workflow-panel">
                <div className="panel-heading"><strong>Project workflow</strong><span>Continue from the next available step.</span></div>
                <div className="workflow-list">
                  <WorkflowRow done={images.length > 0} label="Add data" detail={`${images.length} images available`} />
                  <WorkflowRow done={annotations.some((set) => set.status === "APPROVED")} label="Annotate" detail={`${classes.length} classes configured`} />
                  <WorkflowRow done={datasets.length > 0} label="Create dataset" detail={`${datasets.length} versions`} />
                  <WorkflowRow done={runs.some((run) => run.status === "COMPLETED")} label="Train model" detail={`${runs.length} runs`} />
                  <WorkflowRow done={models.length > 0} label="Test and export" detail={`${models.length} trained models`} />
                </div>
              </section>
            </main>
          )}

          {workspaceSection === "data" && (
            <main className="section-page">
              <PageHeading eyebrow="DATA" title="Images and datasets" description="Import source images, inspect the collection, and prepare dataset versions." />
              <section className="section-panel data-actions-panel">
                <div className="panel-heading"><strong>Source images</strong><span>{images.length} active images</span></div>
                <div className="data-actions">
                  <label className="file-button primary">
                    <Upload size={16} /> Import data
                    <input type="file" accept="image/*,.zip,application/zip" multiple onChange={(event) => { importData(event.target.files); event.currentTarget.value = ""; }} />
                  </label>
                  <button onClick={openImageManager} disabled={!selectedProjectId || busy}><ImageIcon size={16} /> View all images</button>
                </div>
                <div
                  className="source-image-grid data-drop-zone"
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => { event.preventDefault(); importData(event.dataTransfer.files); }}
                >
                  {images.slice(0, 8).map((image) => <button className="source-image-card" key={image.image_id} onClick={() => { void selectAnnotationImage(image.image_id).then((selected) => { if (selected) setWorkspaceSection("annotate"); }); }} type="button"><ProtectedImage image={image} alt={image.original_filename ?? `Image ${image.image_id}`} /><span>{image.original_filename ?? `Image ${image.image_id}`}</span></button>)}
                  {images.length === 0 && <label className="data-empty-drop"><Upload size={25} /><strong>Drop images or a ZIP here</strong><span>Image archives and YOLO datasets are detected automatically.</span><input type="file" accept="image/*,.zip,application/zip" multiple onChange={(event) => { importData(event.target.files); event.currentTarget.value = ""; }} /></label>}
                </div>
              </section>
              <section className="section-panel dataset-section">
                <div className="panel-heading"><strong>Version control</strong><span>View, download, or remove dataset snapshots.</span></div>
                <div className="dataset-version-list">
                  {[...datasets].reverse().map((dataset) => <div className="dataset-version-row" key={dataset.version_id}>
                    <div><strong>Version {dataset.version_number}</strong><span>{dataset.items.length} images · {new Date(dataset.created_at).toLocaleDateString()}</span></div>
                    <button className="icon-button" onClick={() => setDatasetToView(dataset)} title="View version"><Eye size={16} /></button>
                    <button className="icon-button" onClick={() => void download("Download dataset", () => api.downloadDataset(dataset.version_id), `dataset-v${dataset.version_number}.zip`)} disabled={busy} title="Download version"><Download size={16} /></button>
                    <button className="icon-button danger-button" onClick={() => deleteDatasetVersion(dataset)} disabled={busy} title="Delete version"><Trash2 size={16} /></button>
                  </div>)}
                  {datasets.length === 0 && <div className="compact-empty">No dataset versions yet. Create one after labeling images in Annotate.</div>}
                </div>
              </section>
            </main>
          )}

          {workspaceSection === "annotate" && <main className={`workspace annotation-workspace${editableSet ? " is-editing" : ""}`}>
        <aside className="sidebar">
          <SectionTitle title="Classes" />
          <div className="class-form">
            <label>
              Class Name
              <input value={className} onChange={(event) => setClassName(event.target.value)} />
            </label>
            <ClassColorPicker value={classColor} onChange={setClassColor} />
          </div>
          <button onClick={addClass} disabled={!selectedProjectId || busy}><Plus size={16} /> Add Class</button>
          <div className="list">
            {classes.map((labelClass, classIndex) => editingClassId === labelClass.class_id ? (
              <div className="class-edit-row" key={labelClass.class_id}>
                <label>
                  Class name
                  <input value={editingClassName} onChange={(event) => setEditingClassName(event.target.value)} autoFocus maxLength={120} />
                </label>
                <ClassColorPicker value={editingClassColor} onChange={setEditingClassColor} compact />
                <div className="class-edit-actions">
                  <button className="primary" onClick={saveClassEdit} disabled={busy || !editingClassName.trim()} type="button"><Save size={14} /> Save</button>
                  <button onClick={cancelClassEdit} disabled={busy} type="button"><X size={14} /> Cancel</button>
                </div>
              </div>
            ) : (
              <div className={`list-row${labelClass.class_id === selectedClassId ? " active" : ""}${labelClass.class_id === classFilterId ? " filtered" : ""}`} key={labelClass.class_id}>
                <button className="class-select" onClick={() => chooseLabelClass(labelClass.class_id)} aria-pressed={!poseProject && classFilterId === labelClass.class_id} title={!poseProject ? `Filter images by ${labelClass.name}; choose this class for a pending box${classIndex < 9 ? ` (${shortcuts[`class${classIndex + 1}`] || "unassigned"})` : ""}. Press again to clear the filter.` : undefined}>
                  <span className="swatch" style={{ background: labelClass.color }} />
                  <span>{labelClass.class_index}: {labelClass.name}{!poseProject && classIndex < 9 && shortcuts[`class${classIndex + 1}`] ? ` · ${shortcuts[`class${classIndex + 1}`]}` : ""}{classFilterId === labelClass.class_id ? " · Filtering" : ""}</span>
                </button>
                <button className="mini-edit" onClick={() => startClassEdit(labelClass)} title="Edit class" type="button">
                  <Pencil size={14} />
                </button>
                <button className="mini-danger" onClick={() => deleteClass(labelClass)} title="Delete class" type="button">
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
          <div className="annotation-sidebar-tools">
            <SectionTitle title="Annotation" />
            <div className="annotation-image-field">
              <span>Image</span>
              <button onClick={() => setImageSelectorOpen(true)} disabled={images.length === 0 || busy} type="button">
                <ImageIcon size={16} />
                <span>Select image</span>
                <ChevronRight size={15} />
              </button>
              <small>{selectedImage ? `Viewing #${selectedImageIndex} · ${selectedImage.original_filename ?? `image-${selectedImageIndex}`}` : "No image selected"}</small>
            </div>
            <label className="file-button annotation-import-data">
              <Upload size={16} /> Import data
              <input type="file" accept="image/*,.zip,application/zip" multiple disabled={busy} onChange={(event) => { importData(event.target.files, true); event.currentTarget.value = ""; }} />
            </label>
            {poseProject && <button className="annotation-primary-action" onClick={startPoseCollection} disabled={!selectedImageId || selectedClassId == null || busy || !!editableSet || poseCollecting || reviewActionOnCurrentImage} title={`Start a pose (${shortcuts.addPose || "unassigned"}), then click 17 body keypoints`}>
              <Plus size={16} /> Add pose{shortcuts.addPose ? ` · ${shortcuts.addPose}` : ""}
            </button>}
            {!poseProject && pendingClassBox && <div className="annotation-unclassified-hint" role="status">
              <span>Box ready · choose a class above or press its shortcut</span>
              <button type="button" onClick={() => setPendingClassBox(null)}>Discard box</button>
            </div>}
            {selectedImage && annotationImageId === selectedImageId && latestImageAnnotationSet?.is_negative && (
              <div className="annotation-negative-status"><Check size={15} /> Verified negative</div>
            )}
            {selectedImage && annotationImageId === selectedImageId && !latestImageAnnotationSet?.is_negative && (
                <button type="button" onClick={() => confirmNegative(selectedImage.image_id)}
                  disabled={busy || !!editableSet || poseCollecting || drawingBox || !!pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || annotationSaveState === "saving" || reviewActionOnCurrentImage}
                  title={`Confirm this image has no target objects and clear its active labels (${shortcuts.verifyCurrentNegative || "unassigned"})`}>
                  <CircleDot size={16} /> Verify negative{shortcuts.verifyCurrentNegative ? ` · ${shortcuts.verifyCurrentNegative}` : ""}
                </button>
              )}
            {!poseProject && selectedImage && (
              <div className={`annotation-save-state ${annotationSaveState}`} role="status" aria-live="polite">
                <span>{pendingClassBox ? "Choose a class to save this box" : annotationSaveState === "saving" ? `Saving labels in background…${queuedImageCount > 1 ? ` ${queuedImageCount} images queued` : " keep drawing"}` : annotationSaveState === "error" ? "Save failed · labels kept on this image" :
                  draftBoxes.length > 0 ? "Unsaved labels" : annotationSaveState === "saved" ? "Saved" : "Draw a box to label"}</span>
                {(draftBoxes.length > 0 || failedImageBoxes.length > 0) && (
                  <div className="annotation-save-actions">
                    {draftBoxes.length > 0 && <button type="button" onClick={undoDraftBox} disabled={busy} title={`Undo last box (${shortcuts.undoDraft || "unassigned"})`}>Undo</button>}
                    <button type="button" onClick={() => { void saveManualAnnotation(); }} disabled={busy}>
                      {failedImageBoxes.length > 0 ? "Retry" : "Save now"}
                    </button>
                  </div>
                )}
              </div>
            )}
            {poseProject && (poseCollecting || draftBoxes.length > 0 || failedImageBoxes.length > 0) && <div className="annotation-action-grid">
              <button className="primary" onClick={saveManualAnnotation} disabled={(draftBoxes.length === 0 && failedImageBoxes.length === 0) || (poseCollecting && posePoints.length > 0) || busy || !!editableSet} title={`Save completed poses (${shortcuts.save || "unassigned"})`}>
                <Save size={16} /> {failedImageBoxes.length > 0 ? "Retry save" : "Save"} <kbd>{shortcuts.save || "—"}</kbd>
              </button>
              <button onClick={undoDraftBox} disabled={(poseCollecting ? posePoints.length === 0 : draftBoxes.length === 0) || busy || !!editableSet} title={`${poseCollecting ? "Undo last point" : "Undo last unsaved pose"} (${shortcuts.undoAction || shortcuts.undoDraft || "unassigned"})`}>
                <Undo2 size={16} /> Undo <kbd>{shortcuts.undoAction || shortcuts.undoDraft || "—"}</kbd>
              </button>
              <button onClick={() => poseCollecting ? setPosePoints([]) : setDraftBoxes([])} disabled={(poseCollecting ? posePoints.length === 0 : draftBoxes.length === 0) || busy || !!editableSet} title="Clear draft labels">
                <X size={16} /> Clear
              </button>
            </div>}
            {poseCollecting && <div className="pose-sidebar-progress">
              <div><strong>Adding pose</strong><span>{posePoints.length}/17 · {POSE_KEYPOINT_NAMES[posePoints.length] ?? "done"}</span></div>
              <small>Undo {shortcuts.undoAction || shortcuts.undoDraft || "—"} · Cancel {shortcuts.cancelEdit || "—"}</small>
              <button className={poseVisibilityMode === "visible" ? "active" : ""} onClick={() => setPoseVisibilityMode("visible")} type="button">Visible{shortcuts.poseVisible ? ` · ${shortcuts.poseVisible}` : ""}</button>
              <button className={poseVisibilityMode === "occluded" ? "active" : ""} onClick={() => setPoseVisibilityMode("occluded")} type="button">Occluded{shortcuts.poseOccluded ? ` · ${shortcuts.poseOccluded}` : ""}</button>
              <button className="outside-frame-action" onClick={skipPosePoint} type="button">Outside frame{shortcuts.poseOutside ? ` · ${shortcuts.poseOutside}` : ""}</button>
              <button onClick={() => { if (posePoints.length === 0 || window.confirm("Cancel this pose and discard its unfinished points?")) { setPoseCollecting(false); setPosePoints([]); } }} type="button">Cancel{shortcuts.cancelEdit ? ` · ${shortcuts.cancelEdit}` : ""}</button>
            </div>}
          </div>
        </aside>

        <section className="canvas-column">
          {selectedImage && (
            <nav className="annotation-image-navigation" aria-label="Navigate annotation images">
              <button className="icon-button" onClick={() => navigateSelectedImage(-1)} disabled={busy || !adjacentAnnotationImage(-1)} title={`Previous image (${shortcuts.previousImage || "unassigned"})`} aria-label="Previous image" type="button"><ChevronLeft size={20} /></button>
              <button className="annotation-image-position" onClick={() => setImageSelectorOpen(true)} disabled={busy} title="Open image selector" type="button">Image {selectedImageIndex} of {images.length}</button>
              {queuedImageCount > 0 && <span className="annotation-queue-indicator" role="status">Saving {queuedImageCount} image{queuedImageCount === 1 ? "" : "s"}…</span>}
              {queuedAcceptCount > 0 && <span className="annotation-queue-indicator" role="status">Accepting {queuedAcceptCount} image{queuedAcceptCount === 1 ? "" : "s"}…</span>}
              {failedImageCount > 0 && <button className="annotation-queue-error" type="button" onClick={() => {
                const failedImageId = failedBoxesByImage.current.keys().next().value;
                if (failedImageId != null) void selectAnnotationImage(failedImageId);
              }}>Save failed on {failedImageCount} image{failedImageCount === 1 ? "" : "s"} · Open</button>}
              {failedAcceptCount > 0 && <button className="annotation-queue-error" type="button" onClick={() => {
                const failed = [...acceptQueue.current.values()].find((item) => item.state === "failed");
                if (failed) void selectAnnotationImage(failed.set.image_id);
              }}>Accept failed on {failedAcceptCount} image{failedAcceptCount === 1 ? "" : "s"} · Open</button>}
              <button className="annotation-copy-previous" onClick={() => void copyPreviousLabels()} disabled={busy || selectedImageIndex <= 1 || !!editableSet || poseCollecting || !!pendingClassBox || draftBoxes.length > 0 || reviewActionOnCurrentImage || visibleAnnotations.some((annotationSet) => annotationSet.status !== "REJECTED" && annotationSet.annotations.length > 0)} title={selectedImageIndex <= 1 ? "No previous image" : "Copy labels from the previous image as an editable draft"} type="button"><Copy size={17} /> Copy previous labels</button>
              {latestApproved && <button className="annotation-edit-action" onClick={startManualEditor} disabled={busy || !!editableSet || poseCollecting || drawingBox || !!pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || annotationSaveState === "saving" || reviewActionOnCurrentImage} title={`Edit saved labels (${shortcuts.editSavedLabels || "unassigned"})`} type="button"><Settings2 size={16} /> Edit labels <kbd>{shortcuts.editSavedLabels || "—"}</kbd></button>}
              <button className="icon-button" onClick={() => navigateSelectedImage(1)} disabled={busy || !adjacentAnnotationImage(1)} title={`Next image (${shortcuts.nextImage || "unassigned"})`} aria-label="Next image" type="button"><ChevronRight size={20} /></button>
            </nav>
          )}
          {!poseProject && classFilterId != null && <div className="annotation-class-filter-bar" role="status">
            <span className="swatch" style={{ background: classColorById[classFilterId] }} />
            <span>Filter: <strong>{classFilterName}</strong> · {classFilterCount} image{classFilterCount === 1 ? "" : "s"}</span>
            <button type="button" onClick={() => setClassFilterId(null)} aria-label="Clear class filter" title="Clear class filter">Clear <X size={14} /></button>
          </div>}
          <AnnotationCanvas
            image={selectedImage}
            boxes={editableSet ? [] : latestApproved?.annotations ?? []}
            draftBoxes={[...failedImageBoxes, ...queuedImageBoxes, ...draftBoxes]}
            suggestions={editableSet ? [] : pendingSuggestions.flatMap((set) => set.annotations)}
            editableBoxes={editableSet ? reviewBoxes : []}
            selectedEditableIndex={selectedReviewBoxIndex}
            classId={selectedClassId}
            requireClassAfterDraw={!poseProject && !editableSet}
            pendingClassBox={pendingClassBox}
            classColors={classColorById}
            classNames={classNameById}
            onCreateBox={(box) => {
              if (editableSet) addReviewBox(box);
              else {
                setDraftBoxes((current) => [...current, box]);
                if (!poseProject) setAnnotationSaveState("pending");
              }
            }}
            onCreateUnclassifiedBox={setPendingClassBox}
            onDrawingChange={(drawing) => { setDrawingBox(drawing); if (drawing && !poseProject) setPendingClassBox(null); }}
            onEditableSelect={setSelectedReviewBoxIndex}
            onEditableChange={updateReviewBox}
            onEditableEditStart={rememberReviewBoxes}
            onImportData={(files) => importData(files, true)}
            poseCollecting={poseCollecting}
            posePoints={posePoints}
            onPosePoint={addPosePoint}
            poseEditing={poseProject && !!editableSet && selectedReviewBoxIndex != null}
            poseEditingKeypointIndex={selectedPoseKeypointIndex}
            onPoseEditPoint={(point) => updateSelectedPoseKeypoint(point, poseEditVisibilityMode === "visible" ? 2 : 1)}
            busy={busy || imageSwitching || annotationImageId !== selectedImageId || reviewActionOnCurrentImage}
          />
          {pendingSuggestions.length > 0 && <section className="review-band" aria-label="Review queue">
            <SectionTitle title="Review Queue" />
            {pendingSuggestions.map((set) => {
              const acceptItem = acceptQueue.current.get(set.annotation_set_id);
              const reviewBlocked = busy || !!editableSet || !!pendingClassBox || draftBoxes.length > 0 || failedImageBoxes.length > 0 || poseCollecting || annotationSaveState === "saving";
              return <div className="review-row" key={set.annotation_set_id}>
                <span>Set {set.annotation_set_id} · revision {set.revision} · {set.annotations.length} boxes{acceptItem?.state === "queued" ? " · Accept queued" : acceptItem?.state === "failed" ? " · Accept failed" : ""}</span>
                <div>
                  <button onClick={() => { void review(set, "accept"); }} disabled={reviewBlocked || acceptItem?.state === "queued"}><Check size={16} /> {acceptItem?.state === "failed" ? "Retry accept" : "Accept all"}{pendingSuggestions.length === 1 && shortcuts.acceptSuggestion ? ` · ${shortcuts.acceptSuggestion}` : ""}</button>
                  {acceptItem?.state === "failed" && <button onClick={() => discardFailedAccept(set.annotation_set_id)} title={acceptItem.error}>Discard failed action</button>}
                  <button onClick={() => startReviewEditor(set)} disabled={reviewBlocked || !!acceptItem}><Save size={16} /> Edit{pendingSuggestions.length === 1 && shortcuts.editSuggestion ? ` · ${shortcuts.editSuggestion}` : ""}</button>
                  <button onClick={() => { void review(set, "reject"); }} disabled={reviewBlocked || !!acceptItem}><X size={16} /> Reject all{pendingSuggestions.length === 1 && shortcuts.rejectSuggestion ? ` · ${shortcuts.rejectSuggestion}` : ""}</button>
                </div>
              </div>;
            })}
          </section>}
          {!editableSet && draftBoxes.length > 0 && (
            <div className="draft-list">
              {draftBoxes.map((box, index) => {
                const labelClass = classes.find((item) => item.class_id === box.class_id);
                return (
                  <div className="draft-row" key={`${box.class_id}-${box.x_min}-${box.y_min}-${index}`}>
                    <span className="swatch" style={{ background: labelClass?.color ?? "#16a34a" }} />
                    <span>#{index + 1} {labelClass?.name ?? `class ${box.class_id}`}{box.keypoints ? " · 17 keypoints" : ""}</span>
                    <button className="mini-danger" onClick={() => removeDraftBox(index)} title="Delete this box">
                      <Trash2 size={14} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <aside className="rightbar">
          {editableSet && <div className="review-editor">
            <div className="review-editor-summary">
              <strong>{editingManualLabels ? `Edit revision ${editableSet.revision}` : `Review set ${editableSet.annotation_set_id}`}</strong>
              <span>{reviewBoxes.length} boxes · select one on the image</span>
            </div>
            <div className="review-editor-main">
              <label className="review-class-field">Selected class
                <select value={selectedReviewBoxIndex == null ? "" : reviewBoxes[selectedReviewBoxIndex]?.class_id ?? ""} onChange={(event) => changeSelectedReviewClass(Number(event.target.value))} disabled={selectedReviewBoxIndex == null}>
                  <option value="" disabled>Select a box</option>
                  {classes.map((labelClass) => <option key={labelClass.class_id} value={labelClass.class_id}>{labelClass.name}</option>)}
                </select>
              </label>
              {poseProject && reviewBoxes.length > 0 && <div className="pose-editor-controls">
                <label><span>Person</span><select value={selectedReviewBoxIndex ?? ""} onChange={(event) => { setSelectedReviewBoxIndex(Number(event.target.value)); setSelectedPoseKeypointIndex(0); }}>{reviewBoxes.map((_, index) => <option value={index} key={index}>Person {index + 1}</option>)}</select></label>
                <label><span>Keypoint</span><select data-pose-keypoint-select="true" value={selectedPoseKeypointIndex} onChange={(event) => setSelectedPoseKeypointIndex(Number(event.target.value))}>{POSE_KEYPOINT_NAMES.map((name, index) => <option value={index} key={name}>{index + 1}. {name}</option>)}</select></label>
                <div className="pose-visibility-toggle" aria-label="Keypoint visibility">
                  <button className={poseEditVisibilityMode === "visible" ? "active" : ""} onClick={() => setPoseEditVisibilityMode("visible")} type="button">Visible{shortcuts.poseEditVisible ? ` · ${shortcuts.poseEditVisible}` : ""}</button>
                  <button className={poseEditVisibilityMode === "occluded" ? "active" : ""} onClick={() => setPoseEditVisibilityMode("occluded")} type="button">Occluded{shortcuts.poseEditOccluded ? ` · ${shortcuts.poseEditOccluded}` : ""}</button>
                  <button onClick={() => updateSelectedPoseKeypoint(null, 0)} type="button">Outside frame{shortcuts.poseEditOutside ? ` · ${shortcuts.poseEditOutside}` : ""}</button>
                </div>
                <span className="pose-editor-hint">{shortcuts.previousPoint || "—"} previous · {shortcuts.nextPoint || "—"} next. Choose Visible or Occluded, then click the image; Outside frame advances immediately.</span>
              </div>}
              <div className="review-editor-tools">
                <button className="danger-button" onClick={deleteSelectedReviewBox} disabled={selectedReviewBoxIndex == null || busy} title={`Delete selected box (${shortcuts.deleteBox || "unassigned"})`}><Trash2 size={16} /> Delete box</button>
                <button onClick={undoReviewEdit} disabled={reviewHistory.length === 0 || busy} title={`Undo last edit (${shortcuts.undoAction || "unassigned"})`}><Undo2 size={16} /> Undo <kbd>{shortcuts.undoAction || "—"}</kbd></button>
                <button onClick={resetReviewBoxes} disabled={busy} title="Restore labels before editing"><RefreshCw size={16} /> Reset</button>
              </div>
            </div>
            <div className="review-editor-footer">
              <button className="primary" onClick={saveEditedLabels} disabled={busy}><Save size={16} /> {editingManualLabels ? "Save revision" : "Save corrected"} <kbd>{shortcuts.save || "—"}</kbd></button>
              <button onClick={cancelReviewEditor} disabled={busy} title={`Cancel label editing (${shortcuts.cancelEdit || "unassigned"})`}><X size={16} /> Cancel <kbd>{shortcuts.cancelEdit || "—"}</kbd></button>
            </div>
          </div>}
          {!editableSet && <>
          <Panel title="Auto Label">
            <label className="auto-label-model-field">
              <span>Model version</span>
              <select
                value={selectedAutoLabelModelId ?? ""}
                onChange={(event) => setSelectedAutoLabelModelId(
                  event.target.value === "pose-pretrained" ? "pose-pretrained" : event.target.value ? Number(event.target.value) : null,
                )}
                disabled={busy || (!poseProject && models.length === 0)}
              >
                {poseProject && <option value="pose-pretrained">YOLO26 Pose Extra Large pretrained (label assist)</option>}
                {!poseProject && models.length === 0 && <option value="">No model available</option>}
                {[...models].reverse().map((model) => (
                  <option value={model.model_id} key={model.model_id}>
                    Model v{model.version} · {model.source_type === "IMPORTED" ? "Imported" : `mAP50 ${formatMetric(model.map50)}`}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={() => setAutoLabelSettingsOpen(true)} disabled={!selectedAutoLabelModelId || images.length === 0 || busy}><Play size={16} /> {poseProject ? "Generate pose suggestions" : "Run YOLO"}</button>
            <Metric label="Jobs" value={autoJobs.length} />
            <Metric label="Pending" value={pendingReviewImageIds.size} />
          </Panel>
          <Panel title="Dataset">
            <span className="panel-note">Save approved labels as a fixed training snapshot.</span>
            <button className="primary" onClick={() => setDatasetSelectionOpen(true)} disabled={!selectedProjectId || images.length === 0 || busy}><Database size={16} /> Create & Build version</button>
            <Metric label="Versions" value={datasets.length} />
            <Metric label="Latest" value={latestDataset ? `v${latestDataset.version_number}` : "-"} />
          </Panel>
          </>}
        </aside>
        </main>}

        {workspaceSection === "train" && (
          <main className="section-page">
            <PageHeading eyebrow="TRAIN" title="Training runs" description="Choose a dataset, configure the model, and monitor the latest result." action={<button className="primary" onClick={() => setTrainingSetupOpen(true)} disabled={!selectedDataset || busy}><Play size={16} /> Configure & Train</button>} />
            <section className="section-grid">
              <div className="section-panel">
                <div className="panel-heading"><strong>Run configuration</strong><span>The selected dataset and saved training settings.</span></div>
                <label className="auto-label-model-field"><span>Dataset version</span><select value={selectedDataset?.version_id ?? ""} onChange={(event) => setSelectedDatasetVersionId(Number(event.target.value))} disabled={datasets.length === 0 || busy}>{datasets.length === 0 && <option value="">No dataset version</option>}{[...datasets].reverse().map((dataset) => <option value={dataset.version_id} key={dataset.version_id}>Version {dataset.version_number} · {dataset.items.length} images</option>)}</select></label>
                <div className="configuration-grid">
                  <Metric label="Base model" value={formatBaseModel(trainingSettings.baseModel, models)} />
                  <Metric label="Epochs" value={trainingSettings.epochs} />
                  <Metric label="Image size" value={`${trainingSettings.imgsz}px`} />
                  <Metric label="Batch size" value={trainingSettings.batch} />
                </div>
              </div>
              <div className="section-panel">
                <div className="panel-heading"><strong>Latest run</strong><span>{runs.length ? `${runs.length} training runs` : "No training run yet"}</span></div>
                <Metric label="Status" value={runs[runs.length - 1]?.status ?? "-"} />
                <Metric label="Progress" value={`${Math.round(runs[runs.length - 1]?.progress_percent ?? 0)}%`} />
                <Metric label="Models produced" value={models.length} />
                {[...runs].reverse().find((run) => run.can_resume) && (() => {
                  const resumableRun = [...runs].reverse().find((run) => run.can_resume)!;
                  return <div className="training-recovery">
                    <span>Run #{resumableRun.training_run_id} stopped before completion. Its latest checkpoint is available.</span>
                    <button className="primary" onClick={() => resumeYolo(resumableRun.training_run_id)} disabled={busy}><RotateCcw size={16} /> Resume training</button>
                  </div>;
                })()}
              </div>
            </section>
          </main>
        )}

        {workspaceSection === "import-model" && (
          <main className="section-page">
            <PageHeading eyebrow="MODELS / IMPORT" title="Import model" description={`Add trusted YOLO weights to ${selectedProject?.name ?? "this project"} and import their class names automatically.`} />
            <section className="section-panel model-import-panel">
              <div className="panel-heading"><strong>Upload YOLO weights</strong><span>{selectedProjectTask === "POSE_ESTIMATION" ? "17-keypoint Pose Estimation" : "Object Detection"} project</span></div>
              <p>Select an Ultralytics <code>.pt</code> file. The server checks its task and class names, then adds any missing classes to this project. Existing classes must have the same names at the same indexes; nothing is overwritten.</p>
              <label className="model-import-file">
                <Upload size={25} />
                <strong>{modelImportFile?.name ?? "Choose a .pt model file"}</strong>
                <span>{modelImportFile ? `${(modelImportFile.size / (1024 * 1024)).toFixed(1)} MB · ready to import` : "Up to 200 MB · YOLO Detect or YOLO Pose"}</span>
                <input type="file" accept=".pt" disabled={busy} onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setModelImportResult(null);
                  if (file && (!file.name.toLowerCase().endsWith(".pt") || file.size > 200 * 1024 * 1024)) {
                    setModelImportFile(null);
                    setModelImportError(file.size > 200 * 1024 * 1024 ? "Model file must be 200 MB or smaller" : "Choose a .pt model file");
                  } else { setModelImportFile(file); setModelImportError(null); }
                  event.currentTarget.value = "";
                }} />
              </label>
              {modelImportError && <p className="model-import-error" role="alert">{modelImportError}</p>}
              <p className="model-import-warning"><CircleAlert size={17} /> Import only weights from a trusted source. PyTorch .pt checkpoints can contain executable code.</p>
              <button className="primary model-import-submit" type="button" onClick={importModel} disabled={!modelImportFile || busy}><Upload size={17} /> Import into this project</button>
            </section>
            {modelImportResult && <section className="section-panel model-import-success" role="status">
              <Check size={22} />
              <div><strong>Model v{modelImportResult.model.version} imported into {selectedProject?.name ?? "this project"}</strong>
                <span>{modelImportResult.model.class_names?.length ?? 0} classes in model · {modelImportResult.created_classes.length} new classes added</span>
                <p>{modelImportResult.model.class_names?.map((name, index) => `${index}: ${name}`).join(" · ")}</p>
              </div>
              <button type="button" onClick={() => setWorkspaceSection("test")}>Test model <ChevronRight size={16} /></button>
            </section>}
            <section className="section-panel model-import-existing">
              <div className="panel-heading"><strong>Imported in this project</strong><span>{models.filter((model) => model.source_type === "IMPORTED").length} models</span></div>
              {models.filter((model) => model.source_type === "IMPORTED").length === 0 ? <span className="muted">No imported models in this project yet.</span> :
                [...models].reverse().filter((model) => model.source_type === "IMPORTED").map((model) =>
                  <div className="model-import-item" key={model.model_id}><strong>Model v{model.version}</strong><span>{model.model_type === "YOLO26_POSE" ? "Pose" : "Object Detection"} · {model.class_names?.length ?? 0} classes</span><small>{model.class_names?.join(", ")}</small></div>)}
            </section>
          </main>
        )}

        {workspaceSection === "models" && (
          <main className="section-page">
            <PageHeading eyebrow="MODELS / DOWNLOAD" title="Download models" description="Choose a trained or imported model and the format required by your deployment target." />
            <section className="model-version-list">
              {[...models].reverse().map((model) => (
                <article className="section-panel model-version-row" key={model.model_id}>
                  <span className="model-version-icon"><Layers3 size={20} /></span>
                  <div><strong>Model v{model.version}</strong><span>{model.source_type === "IMPORTED" ? "Imported model" : "Trained artifact"}</span></div>
                  <Metric label="mAP50" value={formatMetric(model.map50)} />
                  <label className="model-format-field"><span>Format</span><select value={modelDownloadFormats[model.model_id] ?? "pt"} onChange={(event) => setModelDownloadFormats((current) => ({ ...current, [model.model_id]: event.target.value as ModelDownloadFormat }))}><option value="pt">PyTorch (.pt)</option><option value="onnx">ONNX (.onnx)</option><option value="torchscript">TorchScript (.torchscript)</option></select></label>
                  <div className="model-row-actions" data-model-actions={model.model_id}>
                    <button onClick={() => { const format = modelDownloadFormats[model.model_id] ?? "pt"; void download("Download model", () => api.downloadModel(model.model_id, format), `model-v${model.version}.${format === "torchscript" ? "torchscript" : format}`); }} disabled={busy}><Download size={16} /> Download</button>
                    <div className="model-actions-wrap">
                      <button className="icon-button" onClick={() => setModelActionsOpenId((current) => current === model.model_id ? null : model.model_id)} aria-expanded={modelActionsOpenId === model.model_id} title={`Model v${model.version} actions`} type="button"><MoreHorizontal size={18} /></button>
                      {modelActionsOpenId === model.model_id && (
                        <div className="model-actions-menu">
                          <button className="danger-item" onClick={() => deleteModel(model)} disabled={busy} type="button"><Trash2 size={15} /> Delete model</button>
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              ))}
              {models.length === 0 && <section className="section-panel section-empty"><Layers3 size={26} /><strong>No models yet</strong><span>Train a model or import trusted YOLO weights into this project.</span><button onClick={() => setWorkspaceSection("import-model")}><ChevronRight size={16} /> Import model</button></section>}
            </section>
          </main>
        )}

        {workspaceSection === "test" && <ModelTestWorkspace
          models={models}
          datasets={datasets}
          projectId={selectedProjectId}
          poseProject={poseProject}
          selectedModelId={selectedTestModelId}
          images={testImages}
          imageSource={testDatasetSource}
          predictions={testPredictions}
          failures={testFailures}
          progress={testProgress}
          confidenceThreshold={testConfidenceThreshold}
          classColors={classColorById}
          busy={busy}
          onModelChange={changeTestModel}
          onImagesChange={chooseTestImages}
          onLoadDatasetTest={loadDatasetTestImages}
          onRemoveImage={removeTestImage}
          onClearImages={clearTestImages}
          onThresholdChange={changeTestThreshold}
          onRun={runModelTest}
        />}
          </div>
        </div>
      </div>}

      {newProjectOpen && (
        <NewProjectModal
          busy={busy}
          onClose={() => setNewProjectOpen(false)}
          onCreate={createProject}
        />
      )}

      {autoLabelSettingsOpen && (
        <AutoLabelSettingsModal
          modelName={selectedAutoLabelModelId === "pose-pretrained"
            ? "YOLO26 Pose Extra Large pretrained"
            : `Model v${models.find((model) => model.model_id === selectedAutoLabelModelId)?.version ?? "?"}`}
          imageCount={images.length}
          confidence={autoLabelConfidenceInput}
          onConfidenceChange={setAutoLabelConfidenceInput}
          onClose={() => setAutoLabelSettingsOpen(false)}
          onRun={autoLabel}
        />
      )}

      {autoLabelResults && (
        <AutoLabelResultsModal
          results={autoLabelResults}
          modelName={autoLabelResultsModelName}
          classColors={classColorById}
          onClose={() => setAutoLabelResults(null)}
          onOpen={openAutoLabelResult}
          onAcceptAll={acceptAllAutoLabelResults}
          onRejectAll={rejectAllAutoLabelResults}
          onReviewNegatives={() => { setAutoLabelResults(null); setImageSelectorInitialView("negative"); setImageSelectorOpen(true); }}
          busy={busy}
        />
      )}

      {trainingSetupOpen && (
        <TrainingSetupModal
          settings={trainingSettings}
          models={models}
          datasetVersion={selectedDataset?.version_number ?? null}
          taskType={selectedProjectTask}
          busy={busy}
          onChange={setTrainingSettings}
          onClose={() => setTrainingSetupOpen(false)}
          onTrain={trainYolo}
        />
      )}

      {imageManagerOpen && (
        <ImageManagerModal
          images={managedImages}
          selectedIds={selectedManagedImageIds}
          filter={imageManagerFilter}
          search={imageManagerSearch}
          busy={busy}
          onSelectedIdsChange={setSelectedManagedImageIds}
          onFilterChange={setImageManagerFilter}
          onSearchChange={setImageManagerSearch}
          onArchive={() => updateManagedImagesStatus("ARCHIVED")}
          onRestore={() => updateManagedImagesStatus("UPLOADED")}
          onDeletePermanently={deleteManagedImagesPermanently}
          onClose={() => setImageManagerOpen(false)}
        />
      )}

      {imageSelectorOpen && (
        <ImageSelectorModal
          images={images}
          selectedImageId={selectedImageId}
          classColors={classColorById}
          classNames={classNameById}
          pendingImageIds={pendingReviewImageIds}
          annotationStatuses={imageAnnotationStatuses}
          busy={busy}
          shortcuts={shortcuts}
          initialView={imageSelectorInitialView}
          onSelect={(imageId) => { void selectAnnotationImage(imageId).then((selected) => { if (selected) { setImageSelectorOpen(false); setImageSelectorInitialView("browse"); } }); }}
          onConfirmNegative={confirmNegative}
          onDrawNegative={(imageId) => { void selectAnnotationImage(imageId).then((selected) => { if (selected) { setImageSelectorOpen(false); setImageSelectorInitialView("browse"); } }); }}
          onReviewSelected={reviewSelectedProjectSuggestions}
          onAcceptAll={() => reviewAllProjectSuggestions("accept")}
          onRejectAll={() => reviewAllProjectSuggestions("reject")}
          onClose={() => { setImageSelectorOpen(false); setImageSelectorInitialView("browse"); }}
        />
      )}

      {datasetSelectionOpen && (
        <DatasetSelectionModal
          images={images}
          classColors={classColorById}
          annotationStatuses={imageAnnotationStatuses}
          busy={busy}
          onClose={() => setDatasetSelectionOpen(false)}
          onCreate={createAndBuildDataset}
        />
      )}

      {pendingImportArchive && archiveInspection && (
        <YoloImportModal
          file={pendingImportArchive}
          inspection={archiveInspection}
          limit={importLimit}
          importRemainingImages={importRemainingImages}
          classMode={importClassMode}
          classIndexes={importClassIndexes}
          classNames={importClassNames}
          busy={busy}
          onLimitChange={setImportLimit}
          onImportRemainingImagesChange={setImportRemainingImages}
          onClassModeChange={setImportClassMode}
          onClassIndexesChange={setImportClassIndexes}
          onClassNamesChange={setImportClassNames}
          onImport={confirmYoloImport}
          onClose={() => { setPendingImportArchive(null); setArchiveInspection(null); }}
        />
      )}

      {datasetToView && (
        <DatasetVersionModal
          dataset={datasetToView}
          busy={busy}
          onDownload={() => void download("Download dataset", () => api.downloadDataset(datasetToView.version_id), `dataset-v${datasetToView.version_number}.zip`)}
          onDelete={() => deleteDatasetVersion(datasetToView)}
          onClose={() => setDatasetToView(null)}
        />
      )}

      {operationProgress && (
        <OperationProgressModal
          progress={operationProgress}
          onClose={() => setOperationProgress(null)}
        />
      )}
      {shortcutsOpen && (
        <ShortcutSettingsModal
          bindings={shortcuts}
          onChange={updateShortcuts}
          onClose={() => setShortcutsOpen(false)}
        />
      )}
    </div>
  );
}

function OperationProgressModal({ progress, onClose }: { progress: OperationProgress; onClose: () => void }) {
  const [now, setNow] = useState(Date.now());
  const dialogRef = useRef<HTMLElement>(null);
  useEffect(() => { dialogRef.current?.focus(); }, []);
  useEffect(() => {
    if (progress.error) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [progress.error]);
  const hasCount = progress.total != null && progress.total > 0 && progress.completed != null;
  const completed = hasCount ? Math.min(progress.completed!, progress.total!) : 0;
  const percent = hasCount ? Math.round((completed / progress.total!) * 100) : null;

  return <div className="operation-modal-backdrop">
    <section className="operation-modal" ref={dialogRef} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="operation-modal-title" aria-describedby="operation-modal-detail">
      <div className={progress.error ? "operation-modal-icon error" : "operation-modal-icon"}>
        {progress.error ? <CircleAlert size={25} /> : <RefreshCw size={25} className="operation-modal-spinner" />}
      </div>
      <div className="operation-modal-content">
        <h3 id="operation-modal-title">{progress.error ? `${progress.title} failed` : progress.title}</h3>
        <p id="operation-modal-detail" role="status" aria-live="polite">{progress.error ?? progress.detail}</p>
        {hasCount ? <>
          <div className="operation-modal-count"><span>{completed} / {progress.total} completed</span><strong>{percent}%</strong></div>
          <progress max={progress.total} value={completed} aria-label={`${progress.title} progress`} />
        </> : !progress.error && <progress aria-label={`${progress.title} in progress`} />}
        <span className="operation-modal-time">Elapsed {formatDuration(now - progress.startedAt)}</span>
        {progress.error && <button type="button" className="primary" onClick={onClose}>Close</button>}
      </div>
    </section>
  </div>;
}

function ShortcutSettingsModal({ bindings, onChange, onClose }: {
  bindings: ShortcutBindings;
  onChange: (bindings: ShortcutBindings) => void;
  onClose: () => void;
}) {
  const [recording, setRecording] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const groups = [...new Set(SHORTCUT_DEFINITIONS.map((item) => item.group))];

  useEffect(() => {
    if (recording) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); onClose(); }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, recording]);

  const assign = (id: string, key: string) => {
    const conflict = shortcutConflict(bindings, id, key);
    if (conflict) {
      setError(`${key} is already used by “${conflict.label}” in the same workflow. Change that shortcut first.`);
      return;
    }
    onChange({ ...bindings, [id]: key });
    setRecording(null);
    setError(null);
  };

  const record = (event: ReactKeyboardEvent<HTMLButtonElement>, id: string) => {
    if (recording !== id) return;
    event.preventDefault();
    event.stopPropagation();
    const key = shortcutFromEvent(event);
    if (!key) { setError("Choose a letter, number, arrow, or shortcut combination (not Tab or a modifier alone)."); return; }
    if (key === "Enter" || key === "Space") { setError("Enter and Space activate buttons. Choose another key."); return; }
    if (/^Mod\+(?:R|W|T|N|L|P|Q)$/.test(key)) {
      setError(`${key} is reserved by the browser. Choose another key.`);
      return;
    }
    assign(id, key);
  };

  return <div className="results-modal-backdrop shortcut-settings-backdrop" role="presentation">
    <section className="shortcut-settings-modal" role="dialog" aria-modal="true" aria-labelledby="shortcut-settings-title">
      <header className="shortcut-settings-header">
        <div><h2 id="shortcut-settings-title">Keyboard shortcuts</h2><p>Choose the keys that fit your labeling workflow. Changes are saved for your account on this browser.</p></div>
        <button type="button" className="icon-button" onClick={onClose} title="Close shortcut settings"><X size={20} /></button>
      </header>
      <div className="shortcut-settings-note">Shortcuts work in Annotate when no dialog or text field is active; no-box review shortcuts work inside Select image → Review no-box. Click a key to record a new one. Keys can be shared only when their actions never run in the same workflow.</div>
      {error && <div className="shortcut-settings-error" role="alert">{error}</div>}
      <div className="shortcut-settings-list">
        {groups.map((group) => <section className="shortcut-settings-group" key={group}>
          <h3>{group}</h3>
          {SHORTCUT_DEFINITIONS.filter((item) => item.group === group).map((item) => <div className="shortcut-settings-row" key={item.id}>
            <div><strong>{item.label}</strong><span>{item.description}</span></div>
            <div className="shortcut-settings-controls">
              <button type="button" className={recording === item.id ? "shortcut-recording" : "shortcut-record"}
                aria-label={`Change ${item.label} shortcut`}
                onClick={() => { setRecording(recording === item.id ? null : item.id); setError(null); }}
                onKeyDown={(event) => record(event, item.id)}>
                {recording === item.id ? "Press a key…" : <kbd>{bindings[item.id] || "Unassigned"}</kbd>}
              </button>
              <button type="button" className="shortcut-reset" onClick={() => assign(item.id, item.defaultKey)} disabled={bindings[item.id] === item.defaultKey} title={`Reset ${item.label}`}>Reset</button>
              <button type="button" className="shortcut-reset" onClick={() => assign(item.id, "")} disabled={!bindings[item.id]} title={`Remove ${item.label} shortcut`}>Clear</button>
            </div>
          </div>)}
        </section>)}
      </div>
      <footer className="shortcut-settings-footer"><button type="button" onClick={() => { onChange({ ...DEFAULT_SHORTCUTS }); setRecording(null); setError(null); }}>Reset all to defaults</button><button type="button" className="primary" onClick={onClose}>Done</button></footer>
    </section>
  </div>;
}

type ImageSelectorFilter = "all" | "labeled" | "needs-drawing" | "negative" | "approved" | "pending" | "rejected";

function matchesImageSelectorFilter(filter: ImageSelectorFilter, status: ImageAnnotationStatus | undefined): boolean {
  if (filter === "labeled") return !!status && status.box_count > 0 && status.status !== "REJECTED";
  if (filter === "negative") return status?.is_negative === true;
  if (filter === "approved") return status?.status === "APPROVED";
  if (filter === "pending") return status?.status === "PENDING";
  if (filter === "rejected") return status?.status === "REJECTED";
  if (filter === "needs-drawing") return !status || (status.box_count === 0 && !status.is_negative);
  return true;
}

function filterImageSelectorImages(
  images: ImageItem[],
  filter: ImageSelectorFilter,
  statusByImage: Map<number, ImageAnnotationStatus>,
): ImageItem[] {
  const matching = images.filter((image) => matchesImageSelectorFilter(filter, statusByImage.get(image.image_id)));
  if (filter !== "pending") return matching;

  // Review predictions with boxes first; keep image order within each group.
  const labeled = matching.filter((image) => (statusByImage.get(image.image_id)?.box_count ?? 0) > 0);
  const needsDrawing = matching.filter((image) => (statusByImage.get(image.image_id)?.box_count ?? 0) === 0);
  return [...labeled, ...needsDrawing];
}

function ImageSelectorModal({
  images,
  selectedImageId,
  classColors,
  classNames,
  pendingImageIds,
  annotationStatuses,
  busy,
  shortcuts,
  initialView,
  onSelect,
  onConfirmNegative,
  onDrawNegative,
  onReviewSelected,
  onAcceptAll,
  onRejectAll,
  onClose,
}: {
  images: ImageItem[];
  selectedImageId: number | null;
  classColors: Record<number, string>;
  classNames: Record<number, string>;
  pendingImageIds: Set<number>;
  annotationStatuses: ImageAnnotationStatus[];
  busy: boolean;
  shortcuts: ShortcutBindings;
  initialView: "browse" | "negative";
  onSelect: (imageId: number) => void;
  onConfirmNegative: (imageId: number) => void;
  onDrawNegative: (imageId: number) => void;
  onReviewSelected: (imageIds: number[], mode: "accept" | "reject") => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  onClose: () => void;
}) {
  const batchSize = 24;
  const selectedIndex = Math.max(0, images.findIndex((image) => image.image_id === selectedImageId));
  const initialStart = Math.floor(selectedIndex / batchSize) * batchSize;
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"browse" | "negative">(initialView);
  const [filter, setFilter] = useState<ImageSelectorFilter>("all");
  const [selectingMultiple, setSelectingMultiple] = useState(false);
  const [selectedPendingImageIds, setSelectedPendingImageIds] = useState<Set<number>>(new Set());
  const [rangeStart, setRangeStart] = useState(initialStart);
  const [rangeEnd, setRangeEnd] = useState(Math.min(images.length, initialStart + batchSize));
  const [previews, setPreviews] = useState<Record<number, AnnotationSet | null>>({});
  const dialogRef = useRef<HTMLElement | null>(null);
  const selectedRef = useRef<HTMLButtonElement | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const topSentinelRef = useRef<HTMLDivElement | null>(null);
  const bottomSentinelRef = useRef<HTMLDivElement | null>(null);
  const previousScrollHeightRef = useRef(0);
  const [initialPositionReady, setInitialPositionReady] = useState(false);
  const statusByImage = useMemo(
    () => new Map(annotationStatuses.map((status) => [status.image_id, status])),
    [annotationStatuses],
  );
  const noBoxReviewCount = annotationStatuses.filter((status) =>
    status.source_type === "MODEL" && status.status === "APPROVED" && status.box_count === 0 && !status.is_negative
  ).length;
  const searched = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return images;
    if (/^#?\d+$/.test(query)) {
      const position = Number(query.replace(/^#/, ""));
      return position >= 1 && position <= images.length ? [images[position - 1]] : [];
    }
    return images.filter((image) => (image.original_filename ?? "").toLowerCase().includes(query));
  }, [images, search]);
  const requestedImageNumber = /^#?\d+$/.test(search.trim()) ? Number(search.trim().replace(/^#/, "")) : null;
  const filtered = useMemo(() =>
    filterImageSelectorImages(searched, filter, statusByImage),
    [filter, searched, statusByImage],
  );
  const filterCounts = useMemo(() => ({
    all: images.length,
    labeled: images.filter((image) => { const status = statusByImage.get(image.image_id); return !!status && status.box_count > 0 && status.status !== "REJECTED"; }).length,
    approved: images.filter((image) => statusByImage.get(image.image_id)?.status === "APPROVED").length,
    negative: images.filter((image) => statusByImage.get(image.image_id)?.is_negative).length,
    pending: images.filter((image) => statusByImage.get(image.image_id)?.status === "PENDING").length,
    rejected: images.filter((image) => statusByImage.get(image.image_id)?.status === "REJECTED").length,
    "needs-drawing": images.filter((image) => {
      const status = statusByImage.get(image.image_id);
      return !status || (status.box_count === 0 && !status.is_negative);
    }).length,
  }), [images, statusByImage]);
  const visible = filtered.slice(rangeStart, rangeEnd);
  const visiblePendingImageIds = visible
    .filter((image) => statusByImage.get(image.image_id)?.status === "PENDING")
    .map((image) => image.image_id);
  const allVisiblePendingSelected = visiblePendingImageIds.length > 0
    && visiblePendingImageIds.every((imageId) => selectedPendingImageIds.has(imageId));

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.repeat || busy) return;
      const dialogs = document.querySelectorAll('[role="dialog"][aria-modal="true"]');
      if (dialogs[dialogs.length - 1] !== dialogRef.current) return;
      event.preventDefault();
      if (view === "negative") setView("browse");
      else onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, onClose, view]);

  useEffect(() => {
    let active = true;
    const missing = visible.filter((image) => {
      const status = statusByImage.get(image.image_id);
      if (!Object.prototype.hasOwnProperty.call(previews, image.image_id)) return true;
      return previews[image.image_id]?.annotation_set_id !== status?.annotation_set_id;
    });
    if (missing.length === 0) return () => { active = false; };
    const annotationSetIds = [...new Set(missing.map((image) => statusByImage.get(image.image_id)?.annotation_set_id).filter((id): id is number => id != null))];
    const batches = Array.from({ length: Math.ceil(annotationSetIds.length / 48) }, (_, index) => annotationSetIds.slice(index * 48, (index + 1) * 48));
    void Promise.all(batches.map((ids) => api.listAnnotationPreviews(ids))).then((groups) => {
      if (!active) return;
      const byImage = new Map(groups.flat().map((set) => [set.image_id, set]));
      const entries = missing.map((image) => [image.image_id, byImage.get(image.image_id) ?? null] as const);
      setPreviews((current) => ({ ...current, ...Object.fromEntries(entries) }));
    }).catch(() => undefined);
    return () => { active = false; };
  }, [rangeStart, rangeEnd, filtered, statusByImage]);

  useEffect(() => {
    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        const root = gridRef.current;
        const selected = selectedRef.current;
        if (root && selected) {
          const rootRect = root.getBoundingClientRect();
          const selectedRect = selected.getBoundingClientRect();
          root.scrollTop = Math.max(
            0,
            root.scrollTop + selectedRect.top - rootRect.top - (root.clientHeight - selectedRect.height) / 2,
          );
        }
        setInitialPositionReady(true);
      });
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      window.cancelAnimationFrame(secondFrame);
    };
  }, [selectedImageId, view, filter]);

  useEffect(() => {
    const root = gridRef.current;
    if (!root || !initialPositionReady) return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        if (entry.target === bottomSentinelRef.current && rangeEnd < filtered.length) {
          setRangeEnd((current) => Math.min(filtered.length, current + batchSize));
        }
        if (entry.target === topSentinelRef.current && rangeStart > 0) {
          previousScrollHeightRef.current = root.scrollHeight;
          setRangeStart((current) => Math.max(0, current - batchSize));
        }
      }
    }, { root, rootMargin: "240px 0px", threshold: 0.01 });
    if (topSentinelRef.current) observer.observe(topSentinelRef.current);
    if (bottomSentinelRef.current) observer.observe(bottomSentinelRef.current);
    return () => observer.disconnect();
  }, [filtered.length, initialPositionReady, rangeEnd, rangeStart]);

  useEffect(() => {
    const root = gridRef.current;
    if (!root || previousScrollHeightRef.current === 0) return;
    root.scrollTop += root.scrollHeight - previousScrollHeightRef.current;
    previousScrollHeightRef.current = 0;
  }, [rangeStart]);

  useEffect(() => {
    setSelectedPendingImageIds((current) => new Set(
      [...current].filter((imageId) => statusByImage.get(imageId)?.status === "PENDING"),
    ));
  }, [statusByImage]);

  const updateSearch = (value: string) => {
    setSearch(value);
    if (/^#?\d+$/.test(value.trim())) setFilter("all");
    setRangeStart(0);
    setRangeEnd(Math.min(batchSize, images.length));
  };

  const updateFilter = (value: ImageSelectorFilter) => {
    const nextFiltered = filterImageSelectorImages(searched, value, statusByImage);
    const currentIndex = nextFiltered.findIndex((image) => image.image_id === selectedImageId);
    const start = currentIndex >= 0 ? Math.floor(currentIndex / batchSize) * batchSize : 0;
    setFilter(value);
    setRangeStart(start);
    setRangeEnd(Math.min(nextFiltered.length, start + batchSize));
  };

  const startMultipleSelection = () => {
    setSelectingMultiple(true);
    setSelectedPendingImageIds(new Set());
    updateFilter("pending");
  };

  const stopMultipleSelection = () => {
    setSelectingMultiple(false);
    setSelectedPendingImageIds(new Set());
  };

  const togglePendingImage = (imageId: number) => {
    if (statusByImage.get(imageId)?.status !== "PENDING") return;
    setSelectedPendingImageIds((current) => {
      const next = new Set(current);
      if (next.has(imageId)) next.delete(imageId);
      else next.add(imageId);
      return next;
    });
  };

  const toggleVisiblePendingImages = () => {
    setSelectedPendingImageIds((current) => {
      const next = new Set(current);
      if (allVisiblePendingSelected) visiblePendingImageIds.forEach((imageId) => next.delete(imageId));
      else visiblePendingImageIds.forEach((imageId) => next.add(imageId));
      return next;
    });
  };

  return <div className="results-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section ref={dialogRef} className={`image-selector-modal${view === "negative" ? " review-mode" : ""}`} role="dialog" aria-modal="true" aria-label={view === "negative" ? "Review no-box images" : "Select image"}>
      <header className="image-manager-header">
        {view === "negative" && <button className="icon-button" onClick={() => setView("browse")} title="Back to images" aria-label="Back to images"><ArrowLeft size={18} /></button>}
        <div><h3>{view === "negative" ? "Review no-box images" : "Select image"}</h3><p>{view === "negative" ? "Inspect each image before confirming it contains no objects." : "Choose an image or review model results with no boxes."}</p></div>
        <button className="icon-button" onClick={onClose} title="Close"><X size={18} /></button>
      </header>
      {view === "browse" ? <>
      <div className="image-selector-toolbar">
        <div className="image-selector-search-field">
          <input value={search} onChange={(event) => updateSearch(event.target.value)} placeholder="Image # (e.g. 29) or filename" aria-label="Search by exact image number or filename" />
          <small>Type 29 for image #29 only; words search filenames.</small>
        </div>
        <div className="image-selector-toolbar-meta">
          <span>{filtered.length} images</span>
          {!selectingMultiple && noBoxReviewCount > 0 && <button className="image-selector-negative-action" onClick={() => setView("negative")} disabled={busy} type="button"><CircleDot size={16} /> Review no-box <strong>{noBoxReviewCount}</strong></button>}
          {pendingImageIds.size > 0 && (
            <button className={selectingMultiple ? "active" : ""} onClick={selectingMultiple ? stopMultipleSelection : startMultipleSelection} disabled={busy} type="button">
              <Check size={16} /> {selectingMultiple ? "Done selecting" : "Select multiple"}
            </button>
          )}
        </div>
      </div>
      <div className="image-selector-filters" aria-label="Filter images">
        {([
          ["all", "All"], ["labeled", "Has labels"], ["needs-drawing", "Needs drawing"],
          ["negative", "Verified negative"],
          ["approved", "Accepted"],
          ["pending", "Pending"], ["rejected", "Denied"],
        ] as const).map(([value, label]) => <button className={filter === value ? "active" : ""} onClick={() => updateFilter(value)} type="button" key={value}><span>{label}</span><strong>{filterCounts[value]}</strong></button>)}
      </div>
      <div className="image-selector-grid" ref={gridRef}>
        {filtered.length === 0 && <div className="image-selector-empty">
          {requestedImageNumber != null
            ? requestedImageNumber >= 1 && requestedImageNumber <= images.length
              ? `Image #${requestedImageNumber} is not in this filter. Choose All to find it.`
              : `Image #${requestedImageNumber} not found. Enter a number from 1 to ${images.length}.`
            : search.trim()
              ? "No filenames match. Try a different name or enter an image number."
              : "No images in this filter."}
        </div>}
        {rangeStart > 0 && <div className="image-lazy-sentinel" ref={topSentinelRef}>Loading earlier images...</div>}
        {visible.map((image) => {
          const globalIndex = images.findIndex((item) => item.image_id === image.image_id) + 1;
          const preview = previews[image.image_id];
          const annotationStatus = statusByImage.get(image.image_id);
          const displayStatus = annotationStatus?.status ?? "UNLABELED";
          const hasLabels = (annotationStatus?.box_count ?? 0) > 0;
          const statusText = annotationStatus?.is_negative ? "Verified negative" : displayStatus === "REJECTED"
            ? "Denied"
            : !hasLabels
              ? "Needs drawing"
              : displayStatus === "APPROVED"
                ? `Accepted · ${annotationStatus?.box_count ?? 0}`
                : `Pending · ${annotationStatus?.box_count ?? 0}`;
          const statusClass = annotationStatus?.is_negative ? "negative" : displayStatus === "REJECTED" ? "rejected" : hasLabels ? displayStatus.toLowerCase() : "needs-drawing";
          const tagNames = [...new Set(preview?.annotations.map((box) => classNames[box.class_id] ?? `Class ${box.class_id}`) ?? [])];
          const selected = image.image_id === selectedImageId;
          const bulkSelected = selectedPendingImageIds.has(image.image_id);
          const selectable = displayStatus === "PENDING";
          return <button
            ref={selected ? selectedRef : undefined}
            className={`image-selector-card ${!selectingMultiple && selected ? "active" : ""} ${bulkSelected ? "bulk-selected" : ""} ${selectingMultiple && !selectable ? "bulk-disabled" : ""} ${hasLabels ? "has-labels" : "needs-drawing"}`}
            aria-pressed={selectingMultiple ? bulkSelected : selected}
            aria-disabled={selectingMultiple && !selectable}
            onClick={() => selectingMultiple ? togglePendingImage(image.image_id) : onSelect(image.image_id)}
            key={image.image_id}
            type="button"
          >
            {!selectingMultiple && selected && <span className="image-card-selected"><Check size={14} /> Current image</span>}
            {selectingMultiple && bulkSelected && <span className="image-card-selected"><Check size={14} /> Selected</span>}
            <span className={`image-card-label-state ${hasLabels ? "labeled" : "drawing"}`}>{annotationStatus?.is_negative ? "Negative" : hasLabels ? "Labeled" : "Needs drawing"}</span>
            <ImageSelectorPreview image={image} alt={image.original_filename ?? `Image ${globalIndex}`} preview={preview} classColors={classColors} classNames={classNames} showTags thumbnail scrollRoot={gridRef} />
            <div><strong>#{globalIndex} {image.original_filename ?? `image-${globalIndex}`}</strong><span className={statusClass}>{pendingImageIds.has(image.image_id) && hasLabels ? `Pending · ${annotationStatus?.box_count ?? 0}` : statusText}</span>{tagNames.length > 0 && <small className="image-selector-card-tags" title={tagNames.join(", ")}>Tags: {tagNames.join(", ")}</small>}</div>
          </button>;
        })}
        <div className="image-lazy-sentinel" ref={bottomSentinelRef}>{rangeEnd < filtered.length ? "Loading more images..." : "All images loaded"}</div>
      </div>
      <footer className="image-selector-footer">
        {selectingMultiple ? (
          <div className="image-selector-selection-summary">
            <strong>{selectedPendingImageIds.size} selected</strong>
            <button onClick={toggleVisiblePendingImages} disabled={visiblePendingImageIds.length === 0 || busy} type="button">
              {allVisiblePendingSelected ? "Clear shown" : `Select shown (${visiblePendingImageIds.length})`}
            </button>
            {selectedPendingImageIds.size > 0 && <button onClick={() => setSelectedPendingImageIds(new Set())} disabled={busy} type="button">Clear selection</button>}
          </div>
        ) : <span>Showing {visible.length} of {filtered.length} images · scroll to load more</span>}
        {selectingMultiple ? (
          <div className="image-selector-review-actions">
            <button onClick={() => onReviewSelected([...selectedPendingImageIds], "accept")} disabled={selectedPendingImageIds.size === 0 || busy}><Check size={16} /> Accept selected ({selectedPendingImageIds.size})</button>
            <button className="danger-button" onClick={() => onReviewSelected([...selectedPendingImageIds], "reject")} disabled={selectedPendingImageIds.size === 0 || busy}><Trash2 size={16} /> Reject selected ({selectedPendingImageIds.size})</button>
            <button onClick={stopMultipleSelection} disabled={busy} type="button">Done</button>
          </div>
        ) : pendingImageIds.size > 0 && <div className="image-selector-review-actions">
          <button onClick={onAcceptAll} disabled={busy}><Check size={16} /> Accept all ({pendingImageIds.size})</button>
          <button className="danger-button" onClick={onRejectAll} disabled={busy}><Trash2 size={16} /> Reject all ({pendingImageIds.size})</button>
        </div>}
      </footer>
      </> : <NegativeReviewContent images={images} annotationStatuses={annotationStatuses} busy={busy} shortcuts={shortcuts} onConfirm={onConfirmNegative} onDraw={onDrawNegative} />}
    </section>
  </div>;
}

function YoloImportModal({
  file, inspection, limit, importRemainingImages, classMode, classIndexes, classNames, busy,
  onLimitChange, onImportRemainingImagesChange, onClassModeChange, onClassIndexesChange, onClassNamesChange, onImport, onClose,
}: {
  file: File;
  inspection: ArchiveInspection;
  limit: string;
  importRemainingImages: boolean;
  classMode: "all" | "selected";
  classIndexes: string;
  classNames: string;
  busy: boolean;
  onLimitChange: (value: string) => void;
  onImportRemainingImagesChange: (value: boolean) => void;
  onClassModeChange: (value: "all" | "selected") => void;
  onClassIndexesChange: (value: string) => void;
  onClassNamesChange: (value: string) => void;
  onImport: () => void;
  onClose: () => void;
}) {
  const parsedLimit = Number(limit);
  const importedLabeledImages = limit.trim() !== "" && Number.isFinite(parsedLimit)
    ? Math.max(0, Math.min(inspection.labeled_image_count, Math.floor(parsedLimit)))
    : 0;
  const importedUnlabeledImages = importRemainingImages
    ? inspection.image_count - importedLabeledImages
    : 0;
  const importedImageCount = importedLabeledImages + importedUnlabeledImages;
  const skippedImageCount = inspection.image_count - importedImageCount;
  return <div className="results-modal-backdrop"><section className="training-modal" role="dialog" aria-modal="true" aria-label="Import YOLO dataset">
    <header className="training-modal-header"><div><h3>YOLO dataset detected</h3><p>{file.name} · {inspection.image_count} images</p></div><button className="icon-button" onClick={onClose} disabled={busy}><X size={18} /></button></header>
    <div className="training-form yolo-import-form">
      <section className="yolo-archive-summary training-field-wide" aria-label="Archive summary">
        <div><span>Total images</span><strong>{inspection.image_count}</strong></div>
        <div><span>With labels</span><strong>{inspection.labeled_image_count}</strong></div>
        <div><span>Without labels</span><strong>{inspection.unlabeled_image_count}</strong></div>
      </section>
      <label className="training-field training-field-wide yolo-label-count">
        <span>Images to import with labels</span>
        <div>
          <input min="0" max={inspection.labeled_image_count} type="number" value={limit} onChange={(event) => onLimitChange(event.target.value)} />
          <button type="button" onClick={() => onLimitChange(String(inspection.labeled_image_count))} disabled={busy}>Use all {inspection.labeled_image_count}</button>
        </div>
        <small>{inspection.labeled_image_count} labeled images available</small>
      </label>
      <label className="yolo-remaining-images training-field-wide">
        <input type="checkbox" checked={importRemainingImages} onChange={(event) => onImportRemainingImagesChange(event.target.checked)} />
        <span><strong>Import remaining images without labels</strong><small>{inspection.image_count - importedLabeledImages} images</small></span>
      </label>
      <label className="training-field"><span>YOLO classes</span><select value={classMode} onChange={(event) => onClassModeChange(event.target.value as "all" | "selected")}><option value="all">All classes</option><option value="selected">Selected indexes</option></select></label>
      {classMode === "selected" && <label className="training-field"><span>Class indexes</span><input placeholder="0 or 0,1" value={classIndexes} onChange={(event) => onClassIndexesChange(event.target.value)} /></label>}
      <label className="training-field training-field-wide"><span>Class names</span><input value={classNames} onChange={(event) => onClassNamesChange(event.target.value)} placeholder="Detected from data.yaml" /></label>
      <section className="yolo-import-plan training-field-wide" aria-label="Import plan">
        <strong>{importedImageCount} images will be imported</strong>
        <span>{importedLabeledImages} with labels · {importedUnlabeledImages} without labels{skippedImageCount > 0 ? ` · ${skippedImageCount} skipped` : ""}</span>
      </section>
    </div>
    <footer className="training-modal-footer"><button onClick={onClose} disabled={busy}>Cancel</button><button className="primary" onClick={onImport} disabled={busy || importedImageCount === 0}><Upload size={16} /> Import dataset</button></footer>
  </section></div>;
}

function DatasetVersionModal({ dataset, busy, onDownload, onDelete, onClose }: {
  dataset: DatasetVersion;
  busy: boolean;
  onDownload: () => void;
  onDelete: () => void;
  onClose: () => void;
}) {
  const counts = dataset.items.reduce<Record<string, number>>((result, item) => ({ ...result, [item.split]: (result[item.split] ?? 0) + 1 }), {});
  return <div className="results-modal-backdrop"><section className="dataset-detail-modal" role="dialog" aria-modal="true" aria-label={`Dataset version ${dataset.version_number}`}>
    <header className="training-modal-header"><div><h3>Dataset version {dataset.version_number}</h3><p>Created {new Date(dataset.created_at).toLocaleString()}</p></div><button className="icon-button" onClick={onClose}><X size={18} /></button></header>
    <div className="dataset-detail-body"><Metric label="Images" value={dataset.items.length} /><Metric label="Train" value={counts.TRAIN ?? 0} /><Metric label="Validation" value={counts.VALID ?? 0} /><Metric label="Test" value={counts.TEST ?? 0} /></div>
    <footer className="training-modal-footer"><button className="danger-button" onClick={onDelete} disabled={busy}><Trash2 size={16} /> Delete</button><button className="primary" onClick={onDownload} disabled={busy}><Download size={16} /> Download YOLO ZIP</button></footer>
  </section></div>;
}

function ImageManagerModal({
  images,
  selectedIds,
  filter,
  search,
  busy,
  onSelectedIdsChange,
  onFilterChange,
  onSearchChange,
  onArchive,
  onRestore,
  onDeletePermanently,
  onClose,
}: {
  images: ImageItem[];
  selectedIds: Set<number>;
  filter: ImageManagerFilter;
  search: string;
  busy: boolean;
  onSelectedIdsChange: (ids: Set<number>) => void;
  onFilterChange: (filter: ImageManagerFilter) => void;
  onSearchChange: (search: string) => void;
  onArchive: () => void;
  onRestore: () => void;
  onDeletePermanently: () => void;
  onClose: () => void;
}) {
  const activeCount = images.filter((image) => image.status !== "ARCHIVED").length;
  const archivedCount = images.length - activeCount;
  const normalizedSearch = search.trim().toLowerCase();
  const visibleImages = images.filter((image) => {
    if (filter === "active" && image.status === "ARCHIVED") return false;
    if (filter === "archived" && image.status !== "ARCHIVED") return false;
    if (normalizedSearch && !(image.original_filename ?? "").toLowerCase().includes(normalizedSearch)) return false;
    return true;
  });
  const allVisibleSelected = visibleImages.length > 0 && visibleImages.every((image) => selectedIds.has(image.image_id));
  const selectedActiveCount = images.filter(
    (image) => selectedIds.has(image.image_id) && image.status !== "ARCHIVED",
  ).length;
  const selectedArchivedCount = images.filter(
    (image) => selectedIds.has(image.image_id) && image.status === "ARCHIVED",
  ).length;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose]);

  const toggleImage = (imageId: number) => {
    const next = new Set(selectedIds);
    if (next.has(imageId)) next.delete(imageId);
    else next.add(imageId);
    onSelectedIdsChange(next);
  };

  const toggleAllVisible = () => {
    const next = new Set(selectedIds);
    if (allVisibleSelected) visibleImages.forEach((image) => next.delete(image.image_id));
    else visibleImages.forEach((image) => next.add(image.image_id));
    onSelectedIdsChange(next);
  };

  const filters: { key: ImageManagerFilter; label: string; count: number }[] = [
    { key: "all", label: "All", count: images.length },
    { key: "active", label: "Active", count: activeCount },
    { key: "archived", label: "Archived", count: archivedCount },
  ];

  return (
    <div className="results-modal-backdrop" role="presentation" onMouseDown={() => !busy && onClose()}>
      <section
        className="image-manager-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="image-manager-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="image-manager-header">
          <div>
            <h3 id="image-manager-title">Project images</h3>
            <p>{images.length} images · {selectedIds.size} selected</p>
          </div>
          <button className="icon-button" onClick={onClose} disabled={busy} title="Close image manager"><X size={18} /></button>
        </header>

        <div className="image-manager-controls">
          <div className="image-manager-filters" aria-label="Filter project images">
            {filters.map((item) => (
              <button
                className={filter === item.key ? "active" : ""}
                key={item.key}
                onClick={() => onFilterChange(item.key)}
                aria-pressed={filter === item.key}
              >
                {item.label} <strong>{item.count}</strong>
              </button>
            ))}
          </div>
          <input
            type="search"
            placeholder="Search filename"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
          />
          <button onClick={toggleAllVisible} disabled={visibleImages.length === 0}>
            <Check size={16} /> {allVisibleSelected ? "Clear shown" : `Select shown (${visibleImages.length})`}
          </button>
        </div>

        <div className="image-manager-grid">
          {visibleImages.length === 0 && <div className="results-filter-empty">No images in this view.</div>}
          {visibleImages.map((image) => {
            const selected = selectedIds.has(image.image_id);
            return (
              <button
                className={`image-manager-item${selected ? " selected" : ""}`}
                key={image.image_id}
                onClick={() => toggleImage(image.image_id)}
                aria-pressed={selected}
              >
                <span className="image-manager-check">{selected && <Check size={15} />}</span>
                <ProtectedImage image={image} alt={image.original_filename ?? `Image ${image.image_id}`} />
                <span className={`image-status${image.status === "ARCHIVED" ? " archived" : ""}`}>
                  {image.status === "ARCHIVED" ? "Archived" : "Active"}
                </span>
                <span className="image-manager-name" title={image.original_filename ?? ""}>
                  {image.original_filename ?? `Image ${image.image_id}`}
                </span>
              </button>
            );
          })}
        </div>

        <footer className="image-manager-footer">
          <span>{selectedIds.size} selected</span>
          <div>
            <button onClick={onRestore} disabled={selectedArchivedCount === 0 || busy}>
              <RotateCcw size={16} /> Restore ({selectedArchivedCount})
            </button>
            <button className="danger-button" onClick={onArchive} disabled={selectedActiveCount === 0 || busy}>
              <Archive size={16} /> Archive ({selectedActiveCount})
            </button>
            <button className="danger-button" onClick={onDeletePermanently} disabled={selectedArchivedCount === 0 || busy}>
              <Trash2 size={16} /> Delete permanently ({selectedArchivedCount})
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}

function TrainingSetupModal({
  settings,
  models,
  datasetVersion,
  taskType,
  busy,
  onChange,
  onClose,
  onTrain,
}: {
  settings: YoloTrainingSettings;
  models: ModelItem[];
  datasetVersion: number | null;
  taskType: "OBJECT_DETECTION" | "POSE_ESTIMATION";
  busy: boolean;
  onChange: (settings: YoloTrainingSettings) => void;
  onClose: () => void;
  onTrain: () => void;
}) {
  const poseProject = taskType === "POSE_ESTIMATION";
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const update = <K extends keyof YoloTrainingSettings>(key: K, value: YoloTrainingSettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose]);

  return (
    <div className="results-modal-backdrop" role="presentation" onMouseDown={() => !busy && onClose()}>
      <section
        className="training-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-setup-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="training-modal-header">
          <div>
            <h3 id="training-setup-title">Training setup</h3>
            <p>Dataset version {datasetVersion ?? "-"}</p>
          </div>
          <button className="icon-button" onClick={onClose} disabled={busy} title="Close training setup"><X size={18} /></button>
        </header>

        <div className="training-form">
          <label className="training-field training-field-wide">
            <span>Training task</span>
            <select value={poseProject ? "YOLO26_POSE" : "YOLO26_DETECT"} disabled>
              <option value="YOLO26_DETECT">YOLO26 Object Detection</option>
              <option value="YOLO26_POSE">YOLO26 Pose Estimation</option>
            </select>
          </label>

          <label className="training-field training-field-wide">
            <span>Starting weights</span>
            <select value={settings.baseModel} onChange={(event) => update("baseModel", event.target.value)}>
              <optgroup label={poseProject ? "YOLO26 Pose pretrained" : "YOLO26 pretrained"}>
                {poseProject ? <>
                  <option value="yolo26n-pose.pt">YOLO26 Pose Nano - downloaded</option>
                  <option value="yolo26s-pose.pt">YOLO26 Pose Small</option>
                  <option value="yolo26m-pose.pt">YOLO26 Pose Medium</option>
                  <option value="yolo26l-pose.pt">YOLO26 Pose Large</option>
                  <option value="yolo26x-pose.pt">YOLO26 Pose Extra Large</option>
                </> : <>
                  <option value="yolo26n.pt">YOLO26 Nano - fastest</option>
                  <option value="yolo26s.pt">YOLO26 Small - balanced</option>
                  <option value="yolo26m.pt">YOLO26 Medium</option>
                  <option value="yolo26l.pt">YOLO26 Large</option>
                  <option value="yolo26x.pt">YOLO26 Extra Large</option>
                </>}
              </optgroup>
              {models.length > 0 && (
                <optgroup label="Fine-tune an existing model">
                  {[...models].reverse().map((model) => (
                    <option value={`model:${model.model_id}`} key={model.model_id}>
                      Model v{model.version} · {model.source_type === "IMPORTED" ? "Imported" : `mAP50 ${formatMetric(model.map50)}`}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>

          <NumberField label="Epochs" value={settings.epochs} min={1} max={1000} onChange={(value) => update("epochs", value)} />
          <label className="training-field">
            <span>Image size</span>
            <select value={settings.imgsz} onChange={(event) => update("imgsz", Number(event.target.value))}>
              {[320, 416, 512, 640, 768, 1024].map((size) => <option value={size} key={size}>{size}px</option>)}
            </select>
          </label>
          <label className="training-field">
            <span>Batch size</span>
            <select value={settings.batch} onChange={(event) => update("batch", Number(event.target.value))}>
              <option value={-1}>Auto</option>
              {[1, 2, 4, 8, 16, 32].map((size) => <option value={size} key={size}>{size}</option>)}
            </select>
          </label>
          <label className="training-field">
            <span>Learning rate</span>
            <input type="number" value={settings.lr0} min={0.000001} max={1} step={0.001} onChange={(event) => update("lr0", Number(event.target.value))} />
          </label>

          <button className="training-advanced-toggle training-field-wide" type="button" onClick={() => setAdvancedOpen((current) => !current)} aria-expanded={advancedOpen}>
            <Settings2 size={16} />
            <span>Advanced options</span>
            <ChevronRight size={16} />
          </button>

          {advancedOpen && <div className="training-advanced-panel training-field-wide">
            <NumberField label="Patience" value={settings.patience} min={0} max={500} onChange={(value) => update("patience", value)} />
            <label className="training-field">
              <span>Optimizer</span>
              <select value={settings.optimizer} onChange={(event) => update("optimizer", event.target.value as YoloTrainingSettings["optimizer"])}>
                <option value="auto">Auto</option>
                <option value="AdamW">AdamW</option>
                <option value="Adam">Adam</option>
                <option value="SGD">SGD</option>
              </select>
            </label>
            <label className="training-field">
              <span>Device</span>
              <select value={settings.device} onChange={(event) => update("device", event.target.value as YoloTrainingSettings["device"])}>
                <option value="mps">Apple MPS</option>
                <option value="cpu">CPU</option>
                <option value="auto">Auto</option>
              </select>
            </label>
            <NumberField label="Workers" value={settings.workers} min={0} max={32} onChange={(value) => update("workers", value)} />
            <div className="training-toggles training-field-wide">
              <label><input type="checkbox" checked={settings.amp} onChange={(event) => update("amp", event.target.checked)} /> Mixed precision</label>
              <label><input type="checkbox" checked={settings.cache} onChange={(event) => update("cache", event.target.checked)} /> Cache images</label>
              <label><input type="checkbox" checked={settings.cos_lr} onChange={(event) => update("cos_lr", event.target.checked)} /> Cosine learning rate</label>
            </div>
          </div>}
        </div>

        <footer className="training-modal-footer">
          <button onClick={onClose} disabled={busy}>Cancel</button>
          <button className="primary" onClick={onTrain} disabled={busy || datasetVersion == null}>
            <Play size={16} /> Start training
          </button>
        </footer>
      </section>
    </div>
  );
}

function AutoLabelSettingsModal({
  modelName,
  imageCount,
  confidence,
  onConfidenceChange,
  onClose,
  onRun,
}: {
  modelName: string;
  imageCount: number;
  confidence: string;
  onConfidenceChange: (value: string) => void;
  onClose: () => void;
  onRun: (confidenceThreshold: number) => void;
}) {
  const threshold = Number(confidence);
  const valid = confidence.trim() !== "" && Number.isFinite(threshold) && threshold >= 0.01 && threshold <= 1;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (valid) onRun(threshold);
  };

  return (
    <div className="results-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="auto-label-settings-modal" role="dialog" aria-modal="true" aria-labelledby="auto-label-settings-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="training-modal-header">
          <div><h3 id="auto-label-settings-title">Auto Label settings</h3><p>{modelName} · {imageCount} images in project</p></div>
          <button className="icon-button" type="button" onClick={onClose} title="Close"><X size={18} /></button>
        </header>
        <form onSubmit={submit}>
          <div className="auto-label-settings-body">
            <label className="training-field" htmlFor="auto-label-confidence">
              <span>Confidence threshold (0.01–1.00)</span>
              <input
                id="auto-label-confidence"
                type="number"
                min="0.01"
                max="1"
                step="any"
                inputMode="decimal"
                value={confidence}
                onChange={(event) => onConfidenceChange(event.target.value)}
                autoFocus
                required
              />
            </label>
            <p>Default 0.25 = 25%. A higher threshold usually gives fewer, more confident suggestions; low-confidence objects may be missed. Only unlabeled images are processed.</p>
            {!valid && <span className="auto-label-settings-error">Enter a number from 0.01 to 1.00.</span>}
          </div>
          <footer className="training-modal-footer">
            <button type="button" onClick={onClose}>Cancel</button>
            <button className="primary" type="submit" disabled={!valid}><Play size={16} /> Start Auto Label</button>
          </footer>
        </form>
      </section>
    </div>
  );
}

function NewProjectModal({
  busy,
  onClose,
  onCreate,
}: {
  busy: boolean;
  onClose: () => void;
  onCreate: (details: { name: string; description: string; taskType: "OBJECT_DETECTION" | "POSE_ESTIMATION" }) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<"OBJECT_DETECTION" | "POSE_ESTIMATION">("OBJECT_DETECTION");

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim()) return;
    onCreate({ name, description, taskType });
  };

  return (
    <div className="results-modal-backdrop" role="presentation" onMouseDown={() => !busy && onClose()}>
      <section className="new-project-modal" role="dialog" aria-modal="true" aria-labelledby="new-project-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="training-modal-header">
          <div><h3 id="new-project-title">Create project</h3><p>Choose the annotation and training task for this project.</p></div>
          <button className="icon-button" type="button" onClick={onClose} disabled={busy} title="Close"><X size={18} /></button>
        </header>
        <form className="new-project-form" onSubmit={submit}>
          <label className="training-field"><span>Project name</span><input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} autoFocus required /></label>
          <label className="training-field"><span>Description <small>optional</small></span><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={1000} rows={3} /></label>
          <fieldset className="task-picker"><legend>Project task</legend>
            <label className={taskType === "OBJECT_DETECTION" ? "task-card selected" : "task-card"}>
              <input type="radio" name="task-type" value="OBJECT_DETECTION" checked={taskType === "OBJECT_DETECTION"} onChange={() => setTaskType("OBJECT_DETECTION")} />
              <strong>Object Detection</strong><span>Draw a bounding box around each object. Train with YOLO26 detect weights.</span>
            </label>
            <label className={taskType === "POSE_ESTIMATION" ? "task-card selected" : "task-card"}>
              <input type="radio" name="task-type" value="POSE_ESTIMATION" checked={taskType === "POSE_ESTIMATION"} onChange={() => setTaskType("POSE_ESTIMATION")} />
              <strong>Pose Estimation</strong><span>Start from YOLO26 Pose suggestions, correct 17 human keypoints, then fine-tune your own model.</span>
            </label>
          </fieldset>
          <footer className="training-modal-footer"><button type="button" onClick={onClose} disabled={busy}>Cancel</button><button className="primary" disabled={busy || !name.trim()}><Plus size={16} /> Create project</button></footer>
        </form>
      </section>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="training-field">
      <span>{label}</span>
      <input type="number" value={value} min={min} max={max} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function ClassColorPicker({ value, onChange, compact = false }: {
  value: string;
  onChange: (value: string) => void;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement | null>(null);
  const normalizedValue = value.toLowerCase();
  const customSelected = !DEFAULT_CLASS_COLORS.some((color) => color === normalizedValue);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: MouseEvent) => {
      if (!pickerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", closeOutside);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOutside);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return <div className={`class-color-picker ${compact ? "compact" : ""}`} ref={pickerRef}>
    <span>Color</span>
    <button className="class-color-trigger" type="button" onClick={() => setOpen((current) => !current)} aria-expanded={open} aria-haspopup="dialog">
      <span className="class-color-preview" style={{ background: value }} />
      <span>{value.toUpperCase()}</span>
      <ChevronRight size={15} />
    </button>
    {open && <div className="class-color-popover" role="dialog" aria-label="Choose class color">
      <strong>Choose a color</strong>
      <div className="class-color-swatches">
        {DEFAULT_CLASS_COLORS.map((color) => (
          <button
            aria-label={`Use class color ${color}`}
            aria-pressed={normalizedValue === color}
            className={normalizedValue === color ? "selected" : ""}
            key={color}
            onClick={() => { onChange(color); setOpen(false); }}
            style={{ background: color }}
            title={color}
            type="button"
          >{normalizedValue === color && <Check size={13} />}</button>
        ))}
      </div>
      <label className={customSelected ? "custom-color selected" : "custom-color"}>
        <input type="color" value={value} onChange={(event) => { onChange(event.target.value); setOpen(false); }} />
        <span>Custom color</span>
      </label>
    </div>}
  </div>;
}

function useProtectedImageUrl(image: ImageItem | null, variant: "original" | "thumbnail" = "original") {
  const [url, setUrl] = useState<string | null>(null);
  const imageId = image?.image_id;
  const version = image?.file_hash ?? image?.updated_at ?? image?.created_at;

  useEffect(() => {
    if (!image) {
      setUrl(null);
      return;
    }
    let active = true;
    let objectUrl: string | null = null;
    void api.getImageBlob(image, variant).then((blob) => {
      objectUrl = URL.createObjectURL(blob);
      if (active) setUrl(objectUrl);
      else URL.revokeObjectURL(objectUrl);
    }).catch(() => active && setUrl(null));
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageId, version, variant]);

  return url;
}

function ProtectedImage({ image, className, alt, loading = "lazy" }: {
  image: ImageItem;
  className?: string;
  alt: string;
  loading?: "eager" | "lazy";
}) {
  const url = useProtectedImageUrl(image);
  return url ? <img className={className} src={url} alt={alt} loading={loading} /> : <span className="protected-image-placeholder">Loading image...</span>;
}

function ImageSelectorPreview({
  image,
  alt,
  preview,
  frame,
  classColors,
  classNames,
  showTags = false,
  thumbnail = false,
  scrollRoot,
}: {
  image: ImageItem;
  alt: string;
  preview: AnnotationSet | null | undefined;
  frame?: ImageFrame;
  classColors: Record<number, string>;
  classNames?: Record<number, string>;
  showTags?: boolean;
  thumbnail?: boolean;
  scrollRoot?: { current: HTMLDivElement | null };
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [nearViewport, setNearViewport] = useState(!thumbnail);
  useEffect(() => {
    if (!thumbnail || nearViewport) return;
    const target = containerRef.current;
    if (!target) return;
    if (!window.IntersectionObserver) {
      setNearViewport(true);
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setNearViewport(true);
        observer.disconnect();
      }
    }, { root: scrollRoot?.current ?? null, rootMargin: "300px" });
    observer.observe(target);
    return () => observer.disconnect();
  }, [thumbnail, nearViewport, scrollRoot]);
  const imageUrl = useProtectedImageUrl(thumbnail && !nearViewport ? null : image, thumbnail ? "thumbnail" : "original");
  const [imageReady, setImageReady] = useState(false);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [loadedImageSize, setLoadedImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (frame) return;
    const container = containerRef.current;
    if (!container) return;
    const updateSize = () => setContainerSize({ width: container.clientWidth, height: container.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [frame]);

  useEffect(() => {
    setImageReady(false);
    setLoadedImageSize({ width: 0, height: 0 });
  }, [imageUrl, image.image_id]);

  const visibleFrame = frame ?? (containerSize.width > 0 && containerSize.height > 0 &&
    loadedImageSize.width > 0 && loadedImageSize.height > 0
    ? getImageFrameForDimensions(loadedImageSize.width, loadedImageSize.height, containerSize.width / containerSize.height)
    : null);

  const tone = preview?.status === "PENDING"
    ? "suggestion"
    : preview?.status === "REJECTED"
      ? "rejected"
      : "approved";

  return <div ref={containerRef} className={showTags ? "image-selector-preview show-tags" : "image-selector-preview"}>
    {imageUrl
      ? <img className={imageReady ? undefined : "image-hidden"} src={imageUrl} alt={alt} loading="lazy" onLoad={(event) => {
        setLoadedImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
        setImageReady(true);
      }} />
      : <span className="protected-image-placeholder">Loading image...</span>}
    {imageReady && visibleFrame && preview?.annotations.map((box, index) => <span key={index}>
      <Box box={box} frame={visibleFrame} tone={tone} color={classColors[box.class_id]} label={showTags ? classNames?.[box.class_id] : undefined} />
      <PoseKeypoints keypoints={box.keypoints ?? []} frame={visibleFrame} tone={tone} />
    </span>)}
  </div>;
}

function ModelTestWorkspace({
  models,
  datasets,
  projectId,
  poseProject,
  selectedModelId,
  images,
  imageSource,
  predictions,
  failures,
  progress,
  confidenceThreshold,
  classColors,
  busy,
  onModelChange,
  onImagesChange,
  onLoadDatasetTest,
  onRemoveImage,
  onClearImages,
  onThresholdChange,
  onRun,
}: {
  models: ModelItem[];
  datasets: DatasetVersion[];
  projectId: number | null;
  poseProject: boolean;
  selectedModelId: number | "pose-pretrained" | null;
  images: TestImage[];
  imageSource: string | null;
  predictions: (ModelPrediction | null)[];
  failures: Record<number, string>;
  progress: TestRunProgress;
  confidenceThreshold: number;
  classColors: Record<number, string>;
  busy: boolean;
  onModelChange: (modelId: number | "pose-pretrained" | null) => void;
  onImagesChange: (files: FileList | null) => void;
  onLoadDatasetTest: (versionId: number) => void;
  onRemoveImage: (imageIndex: number) => void;
  onClearImages: () => void;
  onThresholdChange: (value: number) => void;
  onRun: (mode: "untested" | "all") => void;
}) {
  const [visibleTestCount, setVisibleTestCount] = useState(48);
  const [testResultFilter, setTestResultFilter] = useState<TestResultFilter>("all");
  const [isTestFilterMenuOpen, setIsTestFilterMenuOpen] = useState(false);
  const [isFileDragActive, setIsFileDragActive] = useState(false);
  const [chosenDatasetVersionId, setChosenDatasetVersionId] = useState<number | null>(null);
  const testResultsRef = useRef<HTMLDivElement | null>(null);
  const testLoadMoreRef = useRef<HTMLDivElement | null>(null);
  const testFilterMenuRef = useRef<HTMLDivElement | null>(null);
  const lazyBatchSize = 48;
  const projectModels = useMemo(
    () => models.filter((model) => model.project_id === projectId),
    [models, projectId],
  );
  const datasetTestVersions = datasets
    .filter((dataset) => dataset.project_id === projectId && dataset.items.some((item) => item.split === "TEST"))
    .sort((left, right) => left.version_number - right.version_number);
  const activeDatasetVersionId = datasetTestVersions.some((dataset) => dataset.version_id === chosenDatasetVersionId)
    ? chosenDatasetVersionId
    : datasetTestVersions[datasetTestVersions.length - 1]?.version_id ?? null;
  const selectedModel = typeof selectedModelId === "number"
    ? projectModels.find((model) => model.model_id === selectedModelId) ?? null
    : null;
  const completedPredictions = predictions.filter((prediction): prediction is ModelPrediction => prediction != null);
  const totalDetections = completedPredictions.reduce((total, prediction) => total + prediction.boxes.length, 0);
  const detectedImages = completedPredictions.filter((prediction) => prediction.boxes.length > 0).length;
  const noDetectionImages = completedPredictions.filter((prediction) => prediction.boxes.length === 0).length;
  const failedImages = Object.keys(failures).length;
  const confidenceTotal = completedPredictions.reduce(
    (total, prediction) => total + prediction.boxes.reduce((sum, box) => sum + box.confidence, 0),
    0,
  );
  const averageConfidence = totalDetections > 0 ? confidenceTotal / totalDetections : null;
  const progressPercent = progress.total > 0 ? Math.round((progress.processed / progress.total) * 100) : 0;
  const hasTestResults = completedPredictions.length > 0 || failedImages > 0;
  const untestedCount = images.filter((_, index) => predictions[index] == null && failures[index] == null).length;
  const filterCounts: Record<TestResultFilter, number> = {
    all: images.length,
    detected: detectedImages,
    "no-detections": noDetectionImages,
    failed: failedImages,
  };
  const filteredTestImages = images
    .map((image, index) => ({ image, index }))
    .filter(({ index }) => {
      if (testResultFilter === "detected") return (predictions[index]?.boxes.length ?? 0) > 0;
      if (testResultFilter === "no-detections") return predictions[index] != null && predictions[index]?.boxes.length === 0;
      if (testResultFilter === "failed") return failures[index] != null;
      return true;
    });
  const selectedFilterLabel = testResultFilter === "detected"
    ? "Objects found"
    : testResultFilter === "no-detections"
      ? "No objects found"
      : testResultFilter === "failed"
        ? "Failed images"
        : "All images";
  const filteredResultLabel = testResultFilter === "detected"
    ? "images with objects"
    : testResultFilter === "no-detections"
      ? "images with no objects"
      : testResultFilter === "failed"
        ? "failed images"
        : "test images";
  const runButtonLabel = !hasTestResults
    ? `Run ${images.length || ""} test${images.length === 1 ? "" : "s"}`
    : untestedCount > 0
      ? `Run ${untestedCount} new image${untestedCount === 1 ? "" : "s"}`
      : "Run tests again";
  const runTests = () => {
    if (hasTestResults && untestedCount === 0) {
      if (!window.confirm("Run all test images again and replace the current results?")) return;
      onRun("all");
      return;
    }
    onRun(hasTestResults ? "untested" : "all");
  };
  const visibleImages = filteredTestImages.slice(0, visibleTestCount);
  const canHandleFiles = (event: DragEvent<HTMLElement>) => Array.from(event.dataTransfer.types).includes("Files");
  const selectTestResultFilter = (filter: TestResultFilter) => {
    setTestResultFilter(filter);
    setIsTestFilterMenuOpen(false);
    setVisibleTestCount(Math.min(lazyBatchSize, filterCounts[filter]));
    if (testResultsRef.current) testResultsRef.current.scrollTop = 0;
  };

  useEffect(() => {
    setVisibleTestCount((current) => {
      if (filteredTestImages.length === 0) return 0;
      if (current === 0) return Math.min(lazyBatchSize, filteredTestImages.length);
      return Math.min(current, filteredTestImages.length);
    });
  }, [filteredTestImages.length]);

  useEffect(() => {
    const root = testResultsRef.current;
    const sentinel = testLoadMoreRef.current;
    if (!root || !sentinel || visibleTestCount >= filteredTestImages.length) return;
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      setVisibleTestCount((current) => Math.min(filteredTestImages.length, current + lazyBatchSize));
    }, { root, rootMargin: "360px 0px", threshold: 0.01 });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [filteredTestImages.length, visibleTestCount]);
  useEffect(() => {
    const invalidCustomModel = typeof selectedModelId === "number" && selectedModel == null;
    const invalidPoseModel = selectedModelId === "pose-pretrained" && !poseProject;
    if (invalidCustomModel || invalidPoseModel) onModelChange(null);
  }, [onModelChange, poseProject, selectedModel, selectedModelId]);

  useEffect(() => {
    if (testResultFilter === "failed" && failedImages === 0) selectTestResultFilter("all");
  }, [failedImages, testResultFilter]);

  useEffect(() => {
    const closeFilterMenu = (event: MouseEvent) => {
      if (!testFilterMenuRef.current?.contains(event.target as Node)) setIsTestFilterMenuOpen(false);
    };
    document.addEventListener("mousedown", closeFilterMenu);
    return () => document.removeEventListener("mousedown", closeFilterMenu);
  }, []);

  return (
    <main className="test-workspace">
      <section
        className={isFileDragActive ? "test-stage file-drag-active" : "test-stage"}
        onDragEnter={(event) => {
          if (!canHandleFiles(event)) return;
          event.preventDefault();
          setIsFileDragActive(true);
        }}
        onDragOver={(event) => {
          if (!canHandleFiles(event)) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={(event) => {
          if (event.currentTarget.contains(event.relatedTarget as Node)) return;
          setIsFileDragActive(false);
        }}
        onDrop={(event) => {
          if (!canHandleFiles(event)) return;
          event.preventDefault();
          setIsFileDragActive(false);
          onImagesChange(event.dataTransfer.files);
        }}
      >
        <div className="test-controls">
          <div className="test-dataset-source">
            <div>
              <strong>Use saved dataset test split</strong>
              <span>These images are held out from training. Loading them replaces the current test list, but does not change the dataset.</span>
            </div>
            <select
              aria-label="Dataset version for test images"
              value={activeDatasetVersionId ?? ""}
              onChange={(event) => setChosenDatasetVersionId(Number(event.target.value))}
              disabled={busy || datasetTestVersions.length === 0}
            >
              {datasetTestVersions.length === 0 && <option value="">No dataset test split yet</option>}
              {datasetTestVersions.map((dataset) => (
                <option key={dataset.version_id} value={dataset.version_id}>
                  Dataset v{dataset.version_number} · {dataset.items.filter((item) => item.split === "TEST").length} test images
                </option>
              ))}
            </select>
            <button type="button" onClick={() => activeDatasetVersionId != null && onLoadDatasetTest(activeDatasetVersionId)} disabled={busy || activeDatasetVersionId == null}>
              <Database size={16} /> Load test split
            </button>
          </div>
          <div className="test-toolbar">
            <label className="file-button">
            <Upload size={16} /> {images.length > 0 ? "Add images" : "Choose test images"}
              <input
                type="file"
                accept="image/*"
                multiple
                disabled={busy}
              onChange={(event) => {
                onImagesChange(event.target.files);
                event.currentTarget.value = "";
              }}
            />
            </label>
            {images.length > 0 && <button onClick={onClearImages} disabled={busy} type="button"><Trash2 size={16} /> Clear all</button>}
            <label className="test-model-field">
            <span>Model version</span>
            <select
              value={selectedModelId ?? ""}
              onChange={(event) => onModelChange(
                event.target.value === "pose-pretrained"
                  ? "pose-pretrained"
                  : event.target.value ? Number(event.target.value) : null,
              )}
              disabled={busy}
            >
              <option value="">Choose a model</option>
              {poseProject && <option value="pose-pretrained">YOLO26 Pose Nano pretrained (base)</option>}
              {[...projectModels].reverse().map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  Model v{model.version} · {model.source_type === "IMPORTED" ? "Imported" : `mAP50 ${formatMetric(model.map50)}`}
                </option>
              ))}
            </select>
            </label>
            <label className="test-threshold-field">
            <span>Confidence {confidenceThreshold.toFixed(2)}</span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(event) => onThresholdChange(Number(event.target.value))}
              disabled={busy}
            />
            </label>
            <button className="primary" onClick={runTests} disabled={!selectedModelId || images.length === 0 || busy}>
              {hasTestResults && untestedCount === 0 ? <RotateCcw size={16} /> : <Play size={16} />} {runButtonLabel}
            </button>
          </div>
          {imageSource && images.length > 0 && <div className="test-source-caption">Current images: {imageSource} · {images.length} images</div>}
          {images.length > 0 && (
            <div className="test-results-header">
              <div className="test-results-title">
                <strong>Test results</strong>
                <span>{images.length} images · {completedPredictions.length + failedImages} processed</span>
              </div>
              <div className="test-filter-menu" ref={testFilterMenuRef}>
                <button
                  className={isTestFilterMenuOpen ? "test-filter-trigger active" : "test-filter-trigger"}
                  type="button"
                  aria-expanded={isTestFilterMenuOpen}
                  aria-haspopup="menu"
                  onClick={() => setIsTestFilterMenuOpen((open) => !open)}
                >
                  <Filter size={15} /> <span>Filter: {selectedFilterLabel}</span> <ChevronDown size={15} />
                </button>
                {isTestFilterMenuOpen && (
                  <div className="test-filter-options" role="menu">
                    {([
                      ["all", "All images"],
                      ...(completedPredictions.length > 0 ? [
                        ["detected", "Objects found"],
                        ["no-detections", "No objects found"],
                      ] : []),
                      ...(failedImages > 0 ? [["failed", "Failed images"]] : []),
                    ] as [TestResultFilter, string][]).map(([filter, label]) => (
                      <button
                        className={`${testResultFilter === filter ? "active " : ""}${filter === "failed" ? "failed-item" : ""}`.trim()}
                        type="button"
                        role="menuitemradio"
                        aria-checked={testResultFilter === filter}
                        key={filter}
                        onClick={() => selectTestResultFilter(filter)}
                      >
                        <span>{filter === "failed" && <CircleAlert size={14} />}{label}</span>
                        <strong>{filterCounts[filter]}</strong>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        {images.length === 0 ? <div className="test-image-canvas"><div className="empty">Choose test images</div></div>
          : <>
            <div className="test-results-scroll" ref={testResultsRef}>
              <div className="test-results-grid">
                {visibleImages.map(({ image, index: predictionIndex }) => {
                  return <TestImageCard
                    key={image.key}
                    image={image}
                    prediction={predictions[predictionIndex] ?? null}
                    failed={failures[predictionIndex] != null}
                    processing={progress.running && predictionIndex === progress.currentImageIndex}
                    classColors={classColors}
                    busy={busy}
                    onRemove={() => onRemoveImage(predictionIndex)}
                  />;
                })}
                {filteredTestImages.length === 0
                  ? <div className="test-filter-empty">No images in this view</div>
                  : <div className="test-lazy-sentinel" ref={testLoadMoreRef}>
                    {visibleTestCount < filteredTestImages.length ? "Loading more images..." : "All matching images loaded"}
                  </div>}
              </div>
            </div>
            <footer className="test-lazy-status">
              <span>Showing {visibleImages.length} of {filteredTestImages.length} {filteredResultLabel}</span>
              <span>{visibleImages.length < filteredTestImages.length ? "Scroll to load more" : `${images.length} images total`}</span>
            </footer>
          </>}
        {isFileDragActive && <div className="test-drop-overlay"><Upload size={28} /><strong>Drop test images here</strong></div>}
      </section>

      <aside className="rightbar test-sidebar">
        <Panel title="Test run">
          <Metric label="Model" value={selectedModelId === "pose-pretrained" ? "Pose Nano base" : selectedModel ? `v${selectedModel.version}` : "-"} />
          <Metric label="Threshold" value={confidenceThreshold.toFixed(2)} />
          <Metric label="Images" value={images.length} />
          {progress.total > 0 && <div className="test-progress" aria-label={`Processed ${progress.processed} of ${progress.total} images`}>
            <div><strong>{progress.running ? `Processing ${progress.currentFrom}-${progress.currentTo}` : progress.processed > 0 ? "Completed" : "Ready"}</strong><span>{progress.processed} / {progress.total} · {progressPercent}%</span></div>
            <progress max={progress.total} value={progress.processed} />
          </div>}
          <div className="test-summary-grid">
            <Metric label="Detected images" value={progress.processed > 0 ? detectedImages : "-"} />
            <Metric label="No detections" value={progress.processed > 0 ? noDetectionImages : "-"} />
            <Metric label="Failed" value={progress.processed > 0 ? failedImages : "-"} />
            <Metric label="Total detections" value={progress.processed > 0 ? totalDetections : "-"} />
          </div>
          <Metric label="Average confidence" value={averageConfidence == null ? "-" : averageConfidence.toFixed(3)} />
          <Metric label="Elapsed time" value={progress.processed > 0 ? formatDuration(progress.elapsedMs) : "-"} />
        </Panel>
        <Panel title="Predictions">
          {progress.processed === 0 ? <span className="muted">Choose test images and run the selected model.</span>
            : totalDetections === 0 ? <span className="muted">No detections above the selected threshold.</span>
              : <div className="test-prediction-list">
                {predictions.flatMap((prediction, imageIndex) => prediction?.boxes.map((box, boxIndex) => (
                  <div className="test-prediction-row" key={`${imageIndex}-${box.class_id}-${boxIndex}`}>
                    <span className="swatch" style={{ background: classColors[box.class_id] ?? "#16a34a" }} />
                    <strong>{images[imageIndex]?.file.name}: {box.class_name}</strong>
                    <span>{box.confidence.toFixed(2)}</span>
                  </div>
                )) ?? []) }
              </div>}
        </Panel>
      </aside>
    </main>
  );
}

function TestImageCard({
  image,
  prediction,
  failed,
  processing,
  classColors,
  busy,
  onRemove,
}: {
  image: TestImage;
  prediction: ModelPrediction | null;
  failed: boolean;
  processing: boolean;
  classColors: Record<number, string>;
  busy: boolean;
  onRemove: () => void;
}) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const frame = useMemo(
    () => dimensions ? getImageFrameForDimensions(dimensions.width, dimensions.height, 16 / 10) : { left: 0, top: 0, width: 1, height: 1 },
    [dimensions],
  );

  useEffect(() => setDimensions(null), [image.url]);

  return (
    <article className="test-image-card">
      <div className="test-image-canvas">
        <img
          src={image.url}
          alt="Model test input"
          loading="lazy"
          onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
        />
        {dimensions && prediction?.boxes.map((box, index) => (
          <div key={`${box.class_id}-${index}`}>
            <div className="test-box" style={boxStyle(box, frame, classColors[box.class_id])}>
              <span>{box.class_name} {box.confidence.toFixed(2)}</span>
            </div>
            {box.keypoints.length > 0 && <TestPoseSkeleton keypoints={box.keypoints} frame={frame} />}
          </div>
        ))}
      </div>
      <footer>
        <strong title={image.file.name}>{image.file.name}</strong>
        <span className={failed ? "test-result-failed" : ""}>{failed ? "Failed" : prediction ? (prediction.boxes.length > 0 ? `${prediction.boxes.length} detected` : "No detections") : processing ? "Processing" : "Ready"}</span>
        <button className="icon-button test-image-remove" onClick={onRemove} disabled={busy} title={`Remove ${image.file.name}`} type="button"><X size={15} /></button>
      </footer>
    </article>
  );
}

function TestPoseSkeleton({
  keypoints,
  frame,
}: {
  keypoints: { x: number; y: number; visibility: number }[];
  frame: ImageFrame;
}) {
  const pointAt = (index: number) => keypoints[index];
  return (
    <svg
      className="test-pose-skeleton"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      style={{
        left: `${frame.left * 100}%`,
        top: `${frame.top * 100}%`,
        width: `${frame.width * 100}%`,
        height: `${frame.height * 100}%`,
      }}
      aria-label="Pose keypoints"
    >
      {COCO_SKELETON_EDGES.map(([from, to]) => {
        const start = pointAt(from);
        const end = pointAt(to);
        if (!start || !end || start.visibility <= 0 || end.visibility <= 0) return null;
        return <line key={`${from}-${to}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} />;
      })}
      {keypoints.map((point, index) => point.visibility > 0 && (
        <circle key={`${index}-${point.x}-${point.y}`} cx={point.x} cy={point.y} r="0.012" />
      ))}
    </svg>
  );
}

function NegativeReviewContent({ images, annotationStatuses, busy, shortcuts, onConfirm, onDraw }: {
  images: ImageItem[];
  annotationStatuses: ImageAnnotationStatus[];
  busy: boolean;
  shortcuts: ShortcutBindings;
  onConfirm: (imageId: number) => void;
  onDraw: (imageId: number) => void;
}) {
  const candidates = useMemo(() => {
    const noBoxIds = new Set(annotationStatuses.filter((status) =>
      status.source_type === "MODEL" && status.status === "APPROVED" && status.box_count === 0 && !status.is_negative
    ).map((status) => status.image_id));
    return images.filter((image) => noBoxIds.has(image.image_id));
  }, [images, annotationStatuses]);
  const [index, setIndex] = useState(0);
  const image = candidates[Math.min(index, Math.max(0, candidates.length - 1))] ?? null;
  const imageNumber = image ? images.findIndex((item) => item.image_id === image.image_id) + 1 : 0;
  const confirmShortcutInFlight = useRef(false);

  useEffect(() => {
    if (!busy) confirmShortcutInFlight.current = false;
  }, [busy, image?.image_id]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Esc belongs to the parent modal, even if a user mapped another action to it.
      if (event.key === "Escape") return;
      if (busy) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName ?? "")) return;
      if (matchesShortcut(event, shortcuts, "verifyNegative")) {
        event.preventDefault();
        if (!image || event.repeat || confirmShortcutInFlight.current) return;
        confirmShortcutInFlight.current = true;
        onConfirm(image.image_id);
        return;
      }
      if (matchesShortcut(event, shortcuts, "previousImage")) {
        event.preventDefault();
        setIndex((current) => Math.max(0, current - 1));
      }
      if (matchesShortcut(event, shortcuts, "nextImage") && candidates.length > 0) {
        event.preventDefault();
        setIndex((current) => Math.min(candidates.length - 1, current + 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, candidates.length, image, onConfirm, shortcuts]);

  return <div className="image-selector-review-view">
      <div className="negative-review-main">
        {image ? <>
          <div className="negative-review-stage"><ProtectedImage key={image.image_id} image={image} alt={image.original_filename ?? `Image ${imageNumber}`} loading="eager" /></div>
          <aside className="negative-review-details">
            <span className="negative-review-kicker">NEEDS HUMAN REVIEW</span>
            <h4>Image #{imageNumber}</h4>
            <p title={image.original_filename ?? ""}>{image.original_filename ?? `image-${imageNumber}`}</p>
            <div className="negative-review-note"><CircleAlert size={18} /><span>No detection does not mean this image is negative. If an object is visible, draw its box instead.</span></div>
            <button className="primary" type="button" onClick={() => onConfirm(image.image_id)} disabled={busy}><Check size={16} /> Confirm no objects{shortcuts.verifyNegative && <kbd>{shortcuts.verifyNegative}</kbd>}</button>
            <button type="button" onClick={() => onDraw(image.image_id)} disabled={busy}><Pencil size={16} /> Object visible · draw label</button>
          </aside>
        </> : <div className="negative-review-done"><Check size={30} /><strong>No images waiting for negative review</strong><span>Verified negatives can now be selected when you create a dataset version.</span></div>}
      </div>
      <footer className="negative-review-footer">
        <span>{candidates.length} no-box {candidates.length === 1 ? "image" : "images"} awaiting review{image ? ` · ${Math.min(index + 1, candidates.length)} of ${candidates.length}` : ""}</span>
        <div><button type="button" onClick={() => setIndex((current) => Math.max(0, current - 1))} disabled={index === 0 || busy}><ChevronLeft size={16} /> Previous</button><button type="button" onClick={() => setIndex((current) => Math.min(candidates.length - 1, current + 1))} disabled={index >= candidates.length - 1 || busy}>Next <ChevronRight size={16} /></button></div>
      </footer>
  </div>;
}

function DatasetSelectionModal({
  images, classColors, annotationStatuses, busy, onClose, onCreate,
}: {
  images: ImageItem[];
  classColors: Record<number, string>;
  annotationStatuses: ImageAnnotationStatus[];
  busy: boolean;
  onClose: () => void;
  onCreate: (imageIds: number[]) => void;
}) {
  const candidates = useMemo(() => {
    const approved = new Map(annotationStatuses
      .filter((status) => status.status === "APPROVED" && (status.box_count > 0 || status.is_negative))
      .map((status) => [status.image_id, status]));
    return images.flatMap((image) => {
      const status = approved.get(image.image_id);
      return status ? [{ image, status }] : [];
    });
  }, [images, annotationStatuses]);
  const [selectedIds, setSelectedIds] = useState(() => new Set(candidates.map(({ image }) => image.image_id)));
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "manual" | "model" | "negative">("all");
  const [sampleSize, setSampleSize] = useState(200);
  const [visibleCount, setVisibleCount] = useState(24);
  const [dragSelectEnabled, setDragSelectEnabled] = useState(false);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  const dragSelection = useRef<{
    pointerId: number;
    startIndex: number;
    endIndex: number;
    selecting: boolean;
    before: Set<number>;
    x: number;
    y: number;
    frame: number | null;
  } | null>(null);
  const manualCandidates = candidates.filter(({ status }) => status.source_type !== "MODEL" && !status.is_negative);
  const modelCandidates = candidates.filter(({ status }) => status.source_type === "MODEL" && !status.is_negative);
  const filtered = candidates.filter(({ image, status }) => (
    (filter === "all" || (filter === "negative" ? status.is_negative : filter === "manual" ? status.source_type !== "MODEL" && !status.is_negative : status.source_type === "MODEL" && !status.is_negative))
    && (image.original_filename ?? String(image.image_id)).toLowerCase().includes(search.toLowerCase())
  ));
  const visible = filtered.slice(0, visibleCount);
  const selected = candidates.filter(({ image }) => selectedIds.has(image.image_id));
  const selectedManualCount = selected.filter(({ status }) => status.source_type !== "MODEL" && !status.is_negative).length;
  const selectedNegativeCount = selected.filter(({ status }) => status.is_negative).length;
  const allShownSelected = visible.length > 0 && visible.every(({ image }) => selectedIds.has(image.image_id));

  useEffect(() => {
    setVisibleCount(24);
    gridRef.current?.scrollTo({ top: 0 });
  }, [filter, search]);

  useEffect(() => {
    if (!loadMoreRef.current || visibleCount >= filtered.length) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setVisibleCount((count) => Math.min(count + 24, filtered.length));
    }, { root: gridRef.current, rootMargin: "400px" });
    observer.observe(loadMoreRef.current);
    return () => observer.disconnect();
  }, [visibleCount, filtered.length, filter, search]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose]);

  useEffect(() => () => {
    if (dragSelection.current?.frame != null) window.cancelAnimationFrame(dragSelection.current.frame);
  }, []);

  const toggleImage = (imageId: number) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(imageId)) next.delete(imageId);
      else next.add(imageId);
      return next;
    });
  };
  const startDragSelection = (event: PointerEvent<HTMLLabelElement>, index: number) => {
    if (!dragSelectEnabled || busy || event.button !== 0 || (event.target as HTMLElement).closest("input")) return;
    event.preventDefault();
    const grid = gridRef.current;
    if (!grid) return;
    const imageId = filtered[index].image.image_id;
    const selecting = !selectedIds.has(imageId);
    const before = new Set(selectedIds);
    const session = { pointerId: event.pointerId, startIndex: index, endIndex: index,
      selecting, before, x: event.clientX, y: event.clientY, frame: null as number | null };
    dragSelection.current = session;
    setSelectedIds((current) => {
      const next = new Set(current);
      if (selecting) next.add(imageId);
      else next.delete(imageId);
      return next;
    });
    grid.setPointerCapture(event.pointerId);

    const tick = () => {
      if (dragSelection.current !== session) return;
      const bounds = grid.getBoundingClientRect();
      const edge = 60;
      const speed = session.y < bounds.top + edge
        ? -Math.min(24, Math.max(0, bounds.top + edge - session.y) * 0.4)
        : session.y > bounds.bottom - edge
          ? Math.min(24, Math.max(0, session.y - (bounds.bottom - edge)) * 0.4)
          : 0;
      if (speed) grid.scrollTop += speed;
      const hit = document.elementFromPoint(session.x, Math.max(bounds.top + 20, Math.min(bounds.bottom - 20, session.y)));
      let card = hit?.closest<HTMLElement>("[data-dataset-index]");
      if (!card && speed) {
        const inView = Array.from(grid.querySelectorAll<HTMLElement>("[data-dataset-index]"))
          .filter((item) => item.getBoundingClientRect().bottom > bounds.top && item.getBoundingClientRect().top < bounds.bottom);
        card = speed > 0 ? inView[inView.length - 1] : inView[0];
      }
      if (card && grid.contains(card)) {
        const endIndex = Number(card.dataset.datasetIndex);
        if (Number.isInteger(endIndex) && endIndex !== session.endIndex) {
          session.endIndex = endIndex;
          const next = new Set(session.before);
          for (let position = Math.min(session.startIndex, endIndex); position <= Math.max(session.startIndex, endIndex); position += 1) {
            const id = filtered[position]?.image.image_id;
            if (id != null) {
              if (session.selecting) next.add(id);
              else next.delete(id);
            }
          }
          setSelectedIds(next);
        }
      }
      session.frame = window.requestAnimationFrame(tick);
    };
    session.frame = window.requestAnimationFrame(tick);
  };
  const stopDragSelection = (event: PointerEvent<HTMLDivElement>) => {
    const session = dragSelection.current;
    if (!session || session.pointerId !== event.pointerId) return;
    if (session.frame != null) window.cancelAnimationFrame(session.frame);
    dragSelection.current = null;
    if (gridRef.current?.hasPointerCapture(event.pointerId)) gridRef.current.releasePointerCapture(event.pointerId);
  };
  const moveDragSelection = (event: PointerEvent<HTMLDivElement>) => {
    const session = dragSelection.current;
    if (!session || session.pointerId !== event.pointerId) return;
    session.x = event.clientX;
    session.y = event.clientY;
  };
  const addSample = () => {
    const pool = modelCandidates.filter(({ image }) => !selectedIds.has(image.image_id));
    for (let index = pool.length - 1; index > 0; index -= 1) {
      const other = Math.floor(Math.random() * (index + 1));
      [pool[index], pool[other]] = [pool[other], pool[index]];
    }
    const count = Math.max(0, Math.floor(sampleSize));
    setSelectedIds((current) => new Set([...current, ...pool.slice(0, count).map(({ image }) => image.image_id)]));
  };

  return <div className="results-modal-backdrop" onMouseDown={(event) => {
    if (event.target === event.currentTarget && !busy) onClose();
  }}>
    <section className="dataset-selection-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-selection-title">
      <header className="results-modal-header">
        <div><h3 id="dataset-selection-title">Create dataset version</h3><p>{candidates.length} images with approved labels or verified negatives</p></div>
        <button className="icon-button" onClick={onClose} disabled={busy} title="Close"><X size={18} /></button>
      </header>
      <div className="dataset-selection-tools">
        <div className="dataset-selection-actions">
          <button onClick={() => setSelectedIds(new Set(candidates.map(({ image }) => image.image_id)))} disabled={busy}><Check size={16} /> Select all</button>
          <button onClick={() => setSelectedIds(new Set())} disabled={busy}><X size={16} /> Clear</button>
          <button onClick={() => setSelectedIds(new Set(manualCandidates.map(({ image }) => image.image_id)))} disabled={busy || manualCandidates.length === 0}><Pencil size={16} /> Select manual labels ({manualCandidates.length})</button>
          <label className="dataset-sample-size">Auto-label sample<input type="number" min={1} max={modelCandidates.length || 1} value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} disabled={busy} /></label>
          <button onClick={addSample} disabled={busy || !Number.isFinite(sampleSize) || sampleSize < 1 || modelCandidates.every(({ image }) => selectedIds.has(image.image_id))}><Plus size={16} /> Add sample</button>
        </div>
        <div className="dataset-selection-filters">
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search images" aria-label="Search dataset images" />
          <div className="dataset-source-tabs" aria-label="Label source">
            {(["all", "manual", "model", "negative"] as const).map((value) => <button type="button" aria-pressed={filter === value} className={filter === value ? "active" : ""} onClick={() => setFilter(value)} key={value}>{value === "all" ? "All" : value === "manual" ? "Manual labels" : value === "model" ? "Auto labels" : "Verified negative"}</button>)}
          </div>
          <label className="dataset-select-shown"><input type="checkbox" checked={allShownSelected} disabled={busy || visible.length === 0} onChange={() => setSelectedIds((current) => {
            const next = new Set(current);
            visible.forEach(({ image }) => allShownSelected ? next.delete(image.image_id) : next.add(image.image_id));
            return next;
          })} /> Select shown ({visible.length})</label>
          <button type="button" className={dragSelectEnabled ? "active" : ""} aria-pressed={dragSelectEnabled} disabled={busy}
            onClick={() => setDragSelectEnabled((current) => !current)}>Drag select {dragSelectEnabled ? "On" : "Off"}</button>
        </div>
        {dragSelectEnabled && <small className="dataset-drag-hint">Clear first to pick a few frame ranges. Drag across images, release, skip similar frames, then drag again. Start on a selected image to remove a range; hold near an edge to scroll.</small>}
      </div>
      <div className={`dataset-selection-grid${dragSelectEnabled ? " drag-select-enabled" : ""}`} ref={gridRef}
        onPointerMove={moveDragSelection} onPointerUp={stopDragSelection} onPointerCancel={stopDragSelection}>
        {visible.map(({ image, status }, index) => <label className={`dataset-image-item${selectedIds.has(image.image_id) ? " selected" : ""}`} key={image.image_id}
          data-dataset-index={index} onPointerDown={(event) => startDragSelection(event, index)}
          onClick={(event) => { if (dragSelectEnabled && !(event.target as HTMLElement).closest("input")) event.preventDefault(); }}>
          <div className="dataset-image-preview"><DatasetAnnotationPreview image={image} annotationSetId={status.annotation_set_id} classColors={classColors} /><input type="checkbox" checked={selectedIds.has(image.image_id)} onChange={() => toggleImage(image.image_id)} disabled={busy} aria-label={`Include ${image.original_filename ?? image.image_id}`} /></div>
          <div className="dataset-image-meta"><strong title={image.original_filename ?? ""}>{image.original_filename ?? `Image ${image.image_id}`}</strong><span>{status.is_negative ? "Verified negative · 0 boxes" : `${status.source_type === "MODEL" ? "Auto label" : "Manual label"} · ${status.box_count} ${status.box_count === 1 ? "box" : "boxes"}`}</span></div>
        </label>)}
        {filtered.length === 0 && <div className="results-filter-empty">No approved labeled images in this view.</div>}
        {visibleCount < filtered.length && <div className="results-load-more" ref={loadMoreRef}>Loading more images...</div>}
      </div>
      <footer className="results-modal-footer dataset-selection-footer">
        <span>{selected.length} selected · {selectedManualCount} manual · {selected.length - selectedManualCount - selectedNegativeCount} auto · {selectedNegativeCount} negative</span>
        <div className="results-modal-actions"><button onClick={onClose} disabled={busy}>Cancel</button><button className="primary" onClick={() => onCreate(selected.map(({ image }) => image.image_id))} disabled={busy || selected.length === 0}><Database size={16} /> {busy ? "Building..." : `Create & Build (${selected.length})`}</button></div>
      </footer>
    </section>
  </div>;
}

function DatasetAnnotationPreview({ image, annotationSetId, classColors }: {
  image: ImageItem;
  annotationSetId: number;
  classColors: Record<number, string>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [preview, setPreview] = useState<AnnotationSet | null>(null);
  const [failed, setFailed] = useState(false);
  const [inView, setInView] = useState(false);
  const [aspectRatio, setAspectRatio] = useState(4 / 3);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const resize = new ResizeObserver(([entry]) => {
      if (entry.contentRect.height > 0) setAspectRatio(entry.contentRect.width / entry.contentRect.height);
    });
    resize.observe(container);
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setInView(true);
        observer.disconnect();
      }
    }, { root: container.closest(".dataset-selection-grid"), rootMargin: "200px" });
    observer.observe(container);
    return () => { resize.disconnect(); observer.disconnect(); };
  }, []);

  useEffect(() => {
    if (!inView) return;
    let cancelled = false;
    setPreview(null);
    setFailed(false);
    api.listAnnotations(image.image_id).then((sets) => {
      if (cancelled) return;
      const approved = sets.find((set) => set.annotation_set_id === annotationSetId && set.status === "APPROVED");
      setPreview(approved ?? null);
      setFailed(!approved);
    }).catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; };
  }, [inView, image.image_id, annotationSetId]);

  const frame = getImageFrameForDimensions(image.width ?? 1, image.height ?? 1, aspectRatio);
  return <div className="dataset-annotation-preview" ref={containerRef}>
    {inView && <ImageSelectorPreview image={image} alt={image.original_filename ?? `Image ${image.image_id}`} preview={preview} frame={frame} classColors={classColors} />}
    {inView && !preview && <span className="dataset-preview-status">{failed ? "Labels unavailable" : "Loading labels..."}</span>}
  </div>;
}

function AutoLabelResultsModal({
  results,
  modelName,
  classColors,
  onClose,
  onOpen,
  onAcceptAll,
  onRejectAll,
  onReviewNegatives,
  busy,
}: {
  results: AutoLabelResult[];
  modelName: string;
  classColors: Record<number, string>;
  onClose: () => void;
  onOpen: (result: AutoLabelResult) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  onReviewNegatives: () => void;
  busy: boolean;
}) {
  const [resultFilter, setResultFilter] = useState<AutoLabelResultFilter>("all");
  const [visibleCount, setVisibleCount] = useState(24);
  const resultsGridRef = useRef<HTMLDivElement | null>(null);
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  const isAccepted = (result: AutoLabelResult) => result.annotationSet?.status === "APPROVED";
  const activeResults = results.filter((result) => !isAccepted(result));
  const detectedCount = activeResults.filter((result) => (result.annotationSet?.annotations.length ?? 0) > 0).length;
  const emptyCount = results.filter(
    (result) => !isAccepted(result) && result.status === "COMPLETED" && (result.annotationSet?.annotations.length ?? 0) === 0,
  ).length;
  const failedCount = activeResults.filter((result) => result.status === "FAILED").length;
  const acceptedCount = results.filter(isAccepted).length;
  const acceptedNoBoxCount = results.filter((result) => isAccepted(result) && (result.annotationSet?.annotations.length ?? 0) === 0).length;
  const pendingCount = results.filter((result) => result.annotationSet?.status === "PENDING").length;
  const filteredResults = results.filter((result) => {
    const boxCount = result.annotationSet?.annotations.length ?? 0;
    if (resultFilter === "accepted") return isAccepted(result);
    if (isAccepted(result)) return false;
    if (resultFilter === "detected") return boxCount > 0;
    if (resultFilter === "empty") return result.status === "COMPLETED" && boxCount === 0;
    if (resultFilter === "failed") return result.status === "FAILED";
    return true;
  });
  const visibleResults = filteredResults.slice(0, visibleCount);
  const filters: { key: AutoLabelResultFilter; label: string; count: number }[] = [
    { key: "all", label: "All", count: activeResults.length },
    { key: "detected", label: "Detected", count: detectedCount },
    { key: "empty", label: "No boxes", count: emptyCount },
    { key: "failed", label: "Failed", count: failedCount },
    { key: "accepted", label: "Accepted", count: acceptedCount },
  ];

  useEffect(() => {
    setVisibleCount(24);
    resultsGridRef.current?.scrollTo({ top: 0 });
  }, [resultFilter]);

  useEffect(() => {
    if (!loadMoreRef.current || visibleCount >= filteredResults.length) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setVisibleCount((count) => Math.min(count + 24, filteredResults.length));
    }, { root: resultsGridRef.current, rootMargin: "400px" });
    observer.observe(loadMoreRef.current);
    return () => observer.disconnect();
  }, [visibleCount, filteredResults.length, resultFilter]);

  return (
    <div className="results-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="results-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auto-label-results-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="results-modal-header">
          <div>
            <h3 id="auto-label-results-title">Auto-label results</h3>
            <p>{results.length} images processed · {modelName}</p>
          </div>
          <button className="icon-button" onClick={onClose} title="Close results"><X size={18} /></button>
        </header>
        <div className="results-summary" aria-label="Filter auto-label results">
          {filters.map((filter) => (
            <button
              className={`result-filter${resultFilter === filter.key ? " active" : ""}`}
              key={filter.key}
              onClick={() => setResultFilter(filter.key)}
              disabled={filter.key !== "all" && filter.count === 0}
              aria-pressed={resultFilter === filter.key}
            >
              <span>{filter.label}</span>
              <strong>{filter.count}</strong>
            </button>
          ))}
        </div>
        {results.length === 0 ? (
          <div className="results-empty">No images were available for this job.</div>
        ) : (
          <div className="results-grid" ref={resultsGridRef}>
            {filteredResults.length === 0 && (
              <div className="results-filter-empty">No results in this category.</div>
            )}
            {visibleResults.map((result) => {
              const boxCount = result.annotationSet?.annotations.length ?? 0;
              const canReview = result.annotationSet != null && result.annotationSet.status === "PENDING";
              return (
                <article className="result-item" key={result.image.image_id}>
                  <button
                    className="result-preview"
                    onClick={() => onOpen(result)}
                    title={canReview ? "Open this image for review" : "View this image"}
                  >
                    <AutoLabelResultPreview result={result} classColors={classColors} scrollRoot={resultsGridRef} />
                    <span className={boxCount > 0 ? "result-badge detected" : "result-badge"}>
                      {boxCount} {boxCount === 1 ? "box" : "boxes"}
                    </span>
                  </button>
                  <div className="result-item-meta">
                    <span title={result.image.original_filename ?? ""}>
                      {result.image.original_filename ?? `Image ${result.image.image_id}`}
                    </span>
                    <button onClick={() => onOpen(result)}>
                      {canReview ? "Review" : "View"}
                    </button>
                  </div>
                </article>
              );
            })}
            {visibleCount < filteredResults.length && <div className="results-load-more" ref={loadMoreRef}>Loading more images...</div>}
          </div>
        )}
        <div className="results-pager"><span>Showing {visibleResults.length} of {filteredResults.length} images</span></div>
        <footer className="results-modal-footer">
          <span>Select an image to inspect, edit, accept, or reject its predictions.</span>
          <div className="results-modal-actions">
            {acceptedNoBoxCount > 0 && <button type="button" onClick={onReviewNegatives} disabled={busy}><CircleDot size={16} /> Review no-box images ({acceptedNoBoxCount})</button>}
            <button onClick={onAcceptAll} disabled={pendingCount === 0 || busy}>
              <Check size={16} /> Accept all pending ({pendingCount})
            </button>
            <button className="danger-button" onClick={onRejectAll} disabled={pendingCount === 0 || busy}>
              <Trash2 size={16} /> Reject all pending ({pendingCount})
            </button>
            <button className="primary" onClick={onClose} disabled={busy}>Review later</button>
          </div>
        </footer>
      </section>
    </div>
  );
}

function AutoLabelResultPreview({
  result,
  classColors,
  scrollRoot,
}: {
  result: AutoLabelResult;
  classColors: Record<number, string>;
  scrollRoot: { current: HTMLDivElement | null };
}) {
  return <div className="result-preview-content">
    <ImageSelectorPreview
      image={result.image}
      alt={result.image.original_filename ?? `Image ${result.image.image_id}`}
      preview={result.annotationSet}
      classColors={classColors}
      thumbnail
      scrollRoot={scrollRoot}
    />
  </div>;
}

function AnnotationCanvas({
  image,
  boxes,
  draftBoxes,
  suggestions,
  editableBoxes,
  selectedEditableIndex,
  classId,
  requireClassAfterDraw,
  pendingClassBox,
  classColors,
  classNames,
  onCreateBox,
  onCreateUnclassifiedBox,
  onDrawingChange,
  onEditableSelect,
  onEditableChange,
  onEditableEditStart,
  onImportData,
  poseCollecting,
  posePoints,
  onPosePoint,
  poseEditing,
  poseEditingKeypointIndex,
  onPoseEditPoint,
  busy,
}: {
  image: ImageItem | null;
  boxes: AnnotationBox[];
  draftBoxes: DraftBox[];
  suggestions: AnnotationBox[];
  editableBoxes: ReviewBox[];
  selectedEditableIndex: number | null;
  classId: number | null;
  requireClassAfterDraw: boolean;
  pendingClassBox: BoxGeometry | null;
  classColors: Record<number, string>;
  classNames: Record<number, string>;
  onCreateBox: (box: DraftBox) => void;
  onCreateUnclassifiedBox: (box: BoxGeometry) => void;
  onDrawingChange: (drawing: boolean) => void;
  onEditableSelect: (index: number | null) => void;
  onEditableChange: (index: number, box: ReviewBox) => void;
  onEditableEditStart: () => void;
  onImportData: (files: FileList | null) => void;
  poseCollecting: boolean;
  posePoints: PosePoint[];
  onPosePoint: (point: { x: number; y: number }) => void;
  poseEditing: boolean;
  poseEditingKeypointIndex: number;
  onPoseEditPoint: (point: { x: number; y: number }) => void;
  busy: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [current, setCurrent] = useState<{ x: number; y: number } | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [isFileDragActive, setIsFileDragActive] = useState(false);
  const [editDrag, setEditDrag] = useState<{
    index: number;
    mode: EditMode;
    start: { x: number; y: number };
    box: ReviewBox;
  } | null>(null);
  const imageId = image?.image_id ?? null;
  const imageUrl = useProtectedImageUrl(image);
  const imageWidth = naturalSize?.width ?? image?.width ?? 1;
  const imageHeight = naturalSize?.height ?? image?.height ?? 1;
  const imageAspect = imageWidth > 0 && imageHeight > 0 ? imageWidth / imageHeight : 1;
  // The canvas itself follows the image's aspect ratio, so image coordinates fill it exactly.
  const imageFrame: ImageFrame = { left: 0, top: 0, width: 1, height: 1 };

  useEffect(() => {
    setImageReady(false);
    setNaturalSize(null);
    setStart(null);
    setCurrent(null);
    setEditDrag(null);
  }, [imageId]);

  const draft = useMemo(() => {
    if (!start || !current || (!requireClassAfterDraw && classId == null)) return null;
    return { ...normalizeBoxGeometry(start, current), class_id: requireClassAfterDraw ? -1 : classId! };
  }, [classId, current, requireClassAfterDraw, start]);

  const point = (event: PointerEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    const display = imageFrameToPixels(imageFrame, rect.width, rect.height);
    return {
      x: clamp((event.clientX - rect.left - display.left) / display.width),
      y: clamp((event.clientY - rect.top - display.top) / display.height),
    };
  };

  const startEditableDrag = (event: PointerEvent<HTMLDivElement>, index: number, mode: EditMode) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    onEditableSelect(index);
    onEditableEditStart();
    setEditDrag({ index, mode, start: point(event), box: { ...editableBoxes[index] } });
  };

  const moveEditableBox = (event: PointerEvent<HTMLDivElement>) => {
    if (!editDrag) return;
    event.stopPropagation();
    const nextPoint = point(event);
    onEditableChange(
      editDrag.index,
      transformReviewBox(editDrag.box, editDrag.mode, nextPoint.x - editDrag.start.x, nextPoint.y - editDrag.start.y),
    );
  };

  const stopEditableDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (!editDrag) return;
    event.stopPropagation();
    setEditDrag(null);
  };

  const canHandleFiles = (event: DragEvent<HTMLDivElement>) => Array.from(event.dataTransfer.types).includes("Files");

  return (
    <div
      className={`annotation-canvas${image ? " has-image" : ""}${isFileDragActive ? " file-drag-active" : ""}`}
      style={image ? { aspectRatio: `${imageWidth} / ${imageHeight}`, width: `min(100%, ${78 * imageAspect}vh)` } : undefined}
      ref={ref}
      onDragEnter={(event) => {
        if (!canHandleFiles(event)) return;
        event.preventDefault();
        if (!busy) setIsFileDragActive(true);
      }}
      onDragOver={(event) => {
        if (!canHandleFiles(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = busy ? "none" : "copy";
      }}
      onDragLeave={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node)) return;
        setIsFileDragActive(false);
      }}
      onDrop={(event) => {
        if (!canHandleFiles(event)) return;
        event.preventDefault();
        setIsFileDragActive(false);
        if (!busy) onImportData(event.dataTransfer.files);
      }}
      onPointerDown={(event) => {
        if (busy || !imageReady) {
          if (!busy && !image) fileInputRef.current?.click();
          return;
        }
        if (poseCollecting || poseEditing) return;
        onEditableSelect(null);
        event.currentTarget.setPointerCapture(event.pointerId);
        setStart(point(event));
        onDrawingChange(true);
      }}
      onPointerMove={(event) => start && setCurrent(point(event))}
      onPointerUp={(event) => {
        if (busy) {
          setStart(null);
          setCurrent(null);
          onDrawingChange(false);
          return;
        }
        if (poseCollecting && imageReady) {
          onPosePoint(point(event));
          return;
        }
        if (poseEditing && imageReady) {
          onPoseEditPoint(point(event));
          return;
        }
        if (!start || (!requireClassAfterDraw && classId == null)) {
          setStart(null);
          setCurrent(null);
          onDrawingChange(false);
          return;
        }
        const geometry = normalizeBoxGeometry(start, point(event));
        const box = { ...geometry, class_id: classId ?? -1 };
        if (box.x_max - box.x_min > 0.01 && box.y_max - box.y_min > 0.01) {
          if (requireClassAfterDraw) onCreateUnclassifiedBox(geometry);
          else onCreateBox(box);
        }
        setStart(null);
        setCurrent(null);
        onDrawingChange(false);
      }}
      onPointerCancel={() => { setStart(null); setCurrent(null); onDrawingChange(false); }}
    >
      <input
        ref={fileInputRef}
        className="canvas-file-input"
        type="file"
        accept="image/*,.zip,application/zip"
        multiple
        disabled={busy}
        onChange={(event) => {
          onImportData(event.target.files);
          event.currentTarget.value = "";
        }}
      />
      {image ? (
        <>
          {imageUrl ? <img
            className={imageReady ? "" : "image-hidden"}
            src={imageUrl}
            alt={image.original_filename ?? "uploaded"}
            onLoad={(event) => {
              setNaturalSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
              setImageReady(true);
            }}
            onError={() => setImageReady(false)}
          /> : <div className="empty">Loading image...</div>}
          {!imageReady && <div className="empty">Image file missing. Upload this image again.</div>}
        </>
      ) : (
        <div className="empty canvas-upload-empty"><Upload size={24} /><span>Drop images or a ZIP here, or click to browse</span><small>Image archives and YOLO datasets are detected automatically.</small></div>
      )}
      {isFileDragActive && !busy && <div className="canvas-drop-overlay"><Upload size={28} /><strong>Drop images or a ZIP to import</strong></div>}
      {imageReady && boxes.map((box, index) => <Box key={`box-${index}`} box={box} frame={imageFrame} tone="approved" color={classColors[box.class_id]} label={classNames[box.class_id]} />)}
      {imageReady && suggestions.map((box, index) => <Box key={`suggestion-${index}`} box={box} frame={imageFrame} tone="suggestion" color={classColors[box.class_id]} label={classNames[box.class_id]} />)}
      {imageReady && draftBoxes.map((box, index) => <Box key={`draft-${index}`} box={box} frame={imageFrame} tone="draft" color={classColors[box.class_id]} label={classNames[box.class_id]} />)}
      {imageReady && pendingClassBox && <Box box={{ ...pendingClassBox, class_id: -1 }} frame={imageFrame} tone="unclassified" color="#64748b" label="Choose class" />}
      {imageReady && boxes.map((box, index) => <PoseKeypoints key={`saved-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} />)}
      {imageReady && suggestions.map((box, index) => <PoseKeypoints key={`suggestion-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} tone="suggestion" />)}
      {imageReady && draftBoxes.map((box, index) => <PoseKeypoints key={`draft-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} />)}
      {imageReady && poseCollecting && <PoseKeypoints keypoints={posePoints} frame={imageFrame} />}
      {imageReady && editableBoxes.map((box, index) => poseEditing ? (
        <span key={`editable-pose-${index}`}>
          <Box box={box} frame={imageFrame} tone={selectedEditableIndex === index ? "pose-editing selected" : "pose-editing"} color={classColors[box.class_id]} label={classNames[box.class_id]} />
          <PoseKeypoints
            keypoints={box.keypoints ?? []}
            frame={imageFrame}
            selectedIndex={selectedEditableIndex === index ? poseEditingKeypointIndex : null}
          />
        </span>
      ) : (
        <EditableBox
          key={`editable-${index}`}
          box={box}
          frame={imageFrame}
          color={classColors[box.class_id]}
          label={classNames[box.class_id]}
          selected={selectedEditableIndex === index}
          onPointerDown={(event, mode) => startEditableDrag(event, index, mode)}
          onPointerMove={moveEditableBox}
          onPointerUp={stopEditableDrag}
        />
      ))}
      {imageReady && draft && <Box box={draft} frame={imageFrame} tone={requireClassAfterDraw ? "unclassified" : "draft"} color={requireClassAfterDraw ? "#64748b" : classColors[draft.class_id]} label={requireClassAfterDraw ? "Choose class" : classNames[draft.class_id]} />}
    </div>
  );
}

function PoseKeypoints({ keypoints, frame, selectedIndex = null, tone = "approved" }: {
  keypoints: Array<PosePoint | { keypoint_index: number; x: number; y: number; visibility: number }>;
  frame: ImageFrame;
  selectedIndex?: number | null;
  tone?: "approved" | "suggestion" | "rejected";
}) {
  const indexedPoints = new Map(keypoints.map((point, arrayIndex) => [
    "keypoint_index" in point ? point.keypoint_index : arrayIndex,
    point,
  ]));
  return <>
    <svg className={`pose-skeleton ${tone}`} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      {COCO_SKELETON_EDGES.map(([startIndex, endIndex]) => {
        const start = indexedPoints.get(startIndex);
        const end = indexedPoints.get(endIndex);
        if (!start || !end || start.visibility <= 0 || end.visibility <= 0) return null;
        return <line
          key={`${startIndex}-${endIndex}`}
          x1={(frame.left + start.x * frame.width) * 100}
          y1={(frame.top + start.y * frame.height) * 100}
          x2={(frame.left + end.x * frame.width) * 100}
          y2={(frame.top + end.y * frame.height) * 100}
        />;
      })}
    </svg>
    {keypoints.map((point, arrayIndex) => {
    const index = "keypoint_index" in point ? point.keypoint_index : arrayIndex;
    if (point.visibility <= 0) return null;
    return (
    <span
      className={`pose-keypoint ${tone} ${point.visibility === 1 ? "occluded" : ""} ${selectedIndex === index ? "selected" : ""}`}
      key={`${index}-${point.x}-${point.y}`}
      style={{
        left: `${(frame.left + point.x * frame.width) * 100}%`,
        top: `${(frame.top + point.y * frame.height) * 100}%`,
      }}
      title={`${POSE_KEYPOINT_NAMES[index] ?? `keypoint ${index + 1}`} · ${point.visibility === 1 ? "occluded" : "visible"}`}
    >{index + 1}</span>
    );
  })}</>;
}

function EditableBox({
  box,
  frame,
  color,
  label,
  selected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: {
  box: ReviewBox;
  frame: ImageFrame;
  color?: string;
  label?: string;
  selected: boolean;
  onPointerDown: (event: PointerEvent<HTMLDivElement>, mode: EditMode) => void;
  onPointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      className={`box editable ${selected ? "selected" : ""}`}
      style={boxStyle(box, frame, color)}
      onPointerDown={(event) => onPointerDown(event, "move")}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <span className="box-tag" title={label ?? `Class ${box.class_id}`} style={{ borderLeftColor: color ?? "#16a34a" }}>{label ?? `Class ${box.class_id}`}{box.confidence != null ? ` · ${box.confidence.toFixed(2)}` : ""}</span>
      {selected && (["nw", "ne", "sw", "se"] as EditMode[]).map((mode) => (
        <i
          className={`resize-handle ${mode}`}
          key={mode}
          onPointerDown={(event) => onPointerDown(event as unknown as PointerEvent<HTMLDivElement>, mode)}
        />
      ))}
    </div>
  );
}

function Box({
  box,
  frame,
  tone,
  color,
  label,
}: {
  box: Pick<AnnotationBox, "class_id" | "x_min" | "y_min" | "x_max" | "y_max"> & { confidence?: number | null };
  frame: ImageFrame;
  tone: string;
  color?: string;
  label?: string;
}) {
  return (
    <div
      className={`box ${tone}`}
      style={boxStyle(box, frame, color)}
    >
      <span className="box-tag" title={label ?? `Class ${box.class_id}`} style={{ borderLeftColor: color ?? "#16a34a" }}>{label ?? `Class ${box.class_id}`}{box.confidence != null ? ` · ${box.confidence.toFixed(2)}` : ""}</span>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return <div className="panel"><SectionTitle title={title} />{children}</div>;
}

function PageHeading({ eyebrow, title, description, action }: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return <header className="page-heading"><div><span>{eyebrow}</span><h2>{title}</h2><p>{description}</p></div>{action}</header>;
}

function WorkflowRow({ done, label, detail }: { done: boolean; label: string; detail: string }) {
  return <div className="workflow-row"><span className={done ? "done" : ""}>{done ? <Check size={14} /> : <CircleDot size={14} />}</span><div><strong>{label}</strong><small>{detail}</small></div></div>;
}

function SectionTitle({ title }: { title: string }) {
  return <h2>{title}</h2>;
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function toReviewBox(box: AnnotationBox): ReviewBox {
  return {
    class_id: box.class_id,
    x_min: box.x_min,
    y_min: box.y_min,
    x_max: box.x_max,
    y_max: box.y_max,
    confidence: box.confidence,
    keypoints: box.keypoints
      ? [...box.keypoints]
        .sort((left, right) => left.keypoint_index - right.keypoint_index)
        .map(({ x, y, visibility }) => ({ x, y, visibility }))
      : undefined,
  };
}

function cloneReviewBox(box: ReviewBox): ReviewBox {
  return {
    ...box,
    keypoints: box.keypoints?.map((point) => ({ ...point })),
  };
}

function normalizePosePoints(points: PosePoint[] | undefined): PosePoint[] {
  return Array.from({ length: 17 }, (_, index) => {
    const point = points?.[index];
    return point ? { ...point } : { x: 0, y: 0, visibility: 0 };
  });
}

function poseBounds(points: PosePoint[]): Pick<DraftBox, "x_min" | "y_min" | "x_max" | "y_max"> | null {
  const knownPoints = points.filter((point) => point.visibility > 0);
  if (knownPoints.length === 0) return null;
  const padding = 0.02;
  const xValues = knownPoints.map((point) => point.x);
  const yValues = knownPoints.map((point) => point.y);
  const xMin = Math.max(0, Math.min(...xValues) - padding);
  const yMin = Math.max(0, Math.min(...yValues) - padding);
  const xMax = Math.min(1, Math.max(...xValues) + padding);
  const yMax = Math.min(1, Math.max(...yValues) + padding);
  return {
    x_min: xMin,
    y_min: yMin,
    x_max: Math.max(xMin + 0.01, xMax),
    y_max: Math.max(yMin + 0.01, yMax),
  };
}

function resolveEffectiveAnnotationSet(
  annotationSets: AnnotationSet[],
  sourceAnnotationSetId: number | null,
): AnnotationSet | null {
  if (sourceAnnotationSetId == null) return null;
  let current = annotationSets.find((set) => set.annotation_set_id === sourceAnnotationSetId) ?? null;
  const visited = new Set<number>();
  while (current && !visited.has(current.annotation_set_id)) {
    visited.add(current.annotation_set_id);
    const correction = annotationSets
      .filter((set) => (
        set.parent_annotation_set_id === current?.annotation_set_id
        && set.status === "APPROVED"
      ))
      .sort((a, b) => b.revision - a.revision)[0];
    if (!correction) break;
    current = correction;
  }
  return current;
}

function boxStyle(
  box: Pick<AnnotationBox, "x_min" | "y_min" | "x_max" | "y_max">,
  frame: ImageFrame,
  color?: string,
) {
  return {
    left: `${(frame.left + box.x_min * frame.width) * 100}%`,
    top: `${(frame.top + box.y_min * frame.height) * 100}%`,
    width: `${(box.x_max - box.x_min) * frame.width * 100}%`,
    height: `${(box.y_max - box.y_min) * frame.height * 100}%`,
    borderColor: color ?? "#16a34a",
  };
}

function transformReviewBox(box: ReviewBox, mode: EditMode, dx: number, dy: number): ReviewBox {
  const minimumSize = 0.01;
  if (mode === "move") {
    const width = box.x_max - box.x_min;
    const height = box.y_max - box.y_min;
    const xMin = clampRange(box.x_min + dx, 0, 1 - width);
    const yMin = clampRange(box.y_min + dy, 0, 1 - height);
    return { ...box, x_min: xMin, y_min: yMin, x_max: xMin + width, y_max: yMin + height };
  }

  const next = { ...box, confidence: null };
  if (mode.includes("w")) next.x_min = clampRange(box.x_min + dx, 0, box.x_max - minimumSize);
  if (mode.includes("e")) next.x_max = clampRange(box.x_max + dx, box.x_min + minimumSize, 1);
  if (mode.includes("n")) next.y_min = clampRange(box.y_min + dy, 0, box.y_max - minimumSize);
  if (mode.includes("s")) next.y_max = clampRange(box.y_max + dy, box.y_min + minimumSize, 1);
  return next;
}

function normalizeBoxGeometry(a: { x: number; y: number }, b: { x: number; y: number }): BoxGeometry {
  return {
    x_min: Math.min(a.x, b.x),
    y_min: Math.min(a.y, b.y),
    x_max: Math.max(a.x, b.x),
    y_max: Math.max(a.y, b.y),
  };
}

function getImageFrame(image: ImageItem | null): ImageFrame {
  if (!image?.width || !image.height) return { left: 0, top: 0, width: 1, height: 1 };
  return getImageFrameForDimensions(image.width, image.height);
}

function getImageFrameForDimensions(
  width: number,
  height: number,
  canvasAspect = 1,
): ImageFrame {
  const imageAspect = width / height;
  if (imageAspect > canvasAspect) {
    const frameHeight = canvasAspect / imageAspect;
    return { left: 0, top: (1 - frameHeight) / 2, width: 1, height: frameHeight };
  }
  const frameWidth = imageAspect / canvasAspect;
  return { left: (1 - frameWidth) / 2, top: 0, width: frameWidth, height: 1 };
}

function imageFrameToPixels(frame: ImageFrame, containerWidth: number, containerHeight: number) {
  return {
    left: frame.left * containerWidth,
    top: frame.top * containerHeight,
    width: frame.width * containerWidth,
    height: frame.height * containerHeight,
  };
}

function clamp(value: number) {
  return Math.max(0, Math.min(1, value));
}

function clampRange(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Unknown error";
}

function formatMetric(value: number | null) {
  return value == null ? "-" : value.toFixed(3);
}

function formatDuration(milliseconds: number) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  const seconds = milliseconds / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

function formatBaseModel(baseModel: string, models: ModelItem[]) {
  if (!baseModel.startsWith("model:")) return baseModel.replace(".pt", "");
  const modelId = Number(baseModel.slice("model:".length));
  const model = models.find((item) => item.model_id === modelId);
  return model ? `Model v${model.version}` : "Existing model";
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
