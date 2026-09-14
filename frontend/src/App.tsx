import {
  Archive,
  ArrowLeft,
  BarChart3,
  Boxes,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Database,
  Download,
  Eye,
  Image as ImageIcon,
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
import type { DragEvent, FormEvent, PointerEvent, ReactNode } from "react";
import {
  AnnotationBox,
  AnnotationSet,
  api,
  ArchiveInspection,
  AutoLabelJob,
  clearAuthSession,
  DatasetVersion,
  getAuthSession,
  ImageItem,
  ImageAnnotationStatus,
  LabelClass,
  ModelItem,
  ModelDownloadFormat,
  ModelPrediction,
  Project,
  saveAuthSession,
  TrainingRun,
  type AuthSession,
} from "./api";

type DraftBox = {
  class_id: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  keypoints?: PosePoint[];
};

type PosePoint = { x: number; y: number; visibility: number };
type PoseVisibilityMode = "visible" | "occluded";

type ReviewBox = DraftBox & {
  confidence: number | null;
};

type ImageFrame = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type TestImage = {
  key: string;
  file: File;
  url: string;
};

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
type WorkspaceSection = "overview" | "data" | "annotate" | "train" | "models" | "test";

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
  const [authChecked, setAuthChecked] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [imageAnnotationStatuses, setImageAnnotationStatuses] = useState<ImageAnnotationStatus[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationSet[]>([]);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [autoJobs, setAutoJobs] = useState<AutoLabelJob[]>([]);

  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [activeView, setActiveView] = useState<"dashboard" | "workspace">("dashboard");
  const [workspaceSection, setWorkspaceSection] = useState<WorkspaceSection>("annotate");
  const [selectedDatasetVersionId, setSelectedDatasetVersionId] = useState<number | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [projectActionsOpen, setProjectActionsOpen] = useState(false);
  const [className, setClassName] = useState("object");
  const [classColor, setClassColor] = useState("#16a34a");
  const [editingClassId, setEditingClassId] = useState<number | null>(null);
  const [editingClassName, setEditingClassName] = useState("");
  const [editingClassColor, setEditingClassColor] = useState("#16a34a");
  const [importLimit, setImportLimit] = useState("");
  const [importClassMode, setImportClassMode] = useState<"all" | "selected">("all");
  const [importClassIndexes, setImportClassIndexes] = useState("");
  const [importClassNames, setImportClassNames] = useState("");
  const [pendingImportArchive, setPendingImportArchive] = useState<File | null>(null);
  const [archiveInspection, setArchiveInspection] = useState<ArchiveInspection | null>(null);
  const [imageSelectorOpen, setImageSelectorOpen] = useState(false);
  const [datasetToView, setDatasetToView] = useState<DatasetVersion | null>(null);
  const [modelDownloadFormats, setModelDownloadFormats] = useState<Record<number, ModelDownloadFormat>>({});
  const [draftBoxes, setDraftBoxes] = useState<DraftBox[]>([]);
  const [poseCollecting, setPoseCollecting] = useState(false);
  const [posePoints, setPosePoints] = useState<PosePoint[]>([]);
  const [poseVisibilityMode, setPoseVisibilityMode] = useState<PoseVisibilityMode>("visible");
  const [reviewSetId, setReviewSetId] = useState<number | null>(null);
  const [manualEditSetId, setManualEditSetId] = useState<number | null>(null);
  const [reviewBoxes, setReviewBoxes] = useState<ReviewBox[]>([]);
  const [reviewHistory, setReviewHistory] = useState<ReviewBox[][]>([]);
  const [selectedReviewBoxIndex, setSelectedReviewBoxIndex] = useState<number | null>(null);
  const [selectedPoseKeypointIndex, setSelectedPoseKeypointIndex] = useState(0);
  const [poseEditVisibilityMode, setPoseEditVisibilityMode] = useState<PoseVisibilityMode>("visible");
  const [annotationImageId, setAnnotationImageId] = useState<number | null>(null);
  const [autoLabelResults, setAutoLabelResults] = useState<AutoLabelResult[] | null>(null);
  const [reviewToOpen, setReviewToOpen] = useState<AnnotationSet | null>(null);
  const [trainingSetupOpen, setTrainingSetupOpen] = useState(false);
  const [trainingSettings, setTrainingSettings] = useState<YoloTrainingSettings>(defaultTrainingSettings);
  const [selectedAutoLabelModelId, setSelectedAutoLabelModelId] = useState<number | "pose-pretrained" | null>(null);
  const [selectedTestModelId, setSelectedTestModelId] = useState<number | "pose-pretrained" | null>(null);
  const [testImages, setTestImages] = useState<TestImage[]>([]);
  const [testPredictions, setTestPredictions] = useState<ModelPrediction[]>([]);
  const [testConfidenceThreshold, setTestConfidenceThreshold] = useState(0.25);
  const [imageManagerOpen, setImageManagerOpen] = useState(false);
  const [managedImages, setManagedImages] = useState<ImageItem[]>([]);
  const [selectedManagedImageIds, setSelectedManagedImageIds] = useState<Set<number>>(new Set());
  const [imageManagerFilter, setImageManagerFilter] = useState<ImageManagerFilter>("all");
  const [imageManagerSearch, setImageManagerSearch] = useState("");
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const projectRequestId = useRef(0);
  const projectActionsRef = useRef<HTMLDivElement | null>(null);

  const selectedImage = images.find((image) => image.image_id === selectedImageId) ?? null;
  const selectedProject = projects.find((project) => project.project_id === selectedProjectId) ?? null;
  const selectedProjectTask = selectedProject?.task_type === "POSE_ESTIMATION" ? "POSE_ESTIMATION" : "OBJECT_DETECTION";
  const poseProject = selectedProjectTask === "POSE_ESTIMATION";
  const visibleAnnotations = annotationImageId === selectedImageId ? annotations : [];
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
  const pendingReviewImageIds = useMemo(() => new Set(
    autoJobs.flatMap((job) => job.items)
      .filter((item) => item.status === "COMPLETED" && item.annotation_set_id != null)
      .map((item) => item.image_id),
  ), [autoJobs]);
  const pendingProjectAnnotationSetIds = useMemo(() => {
    const pendingImages = new Set(
      imageAnnotationStatuses.filter((status) => status.status === "PENDING").map((status) => status.image_id),
    );
    return [...new Set(autoJobs.flatMap((job) => job.items)
      .filter((item) => item.annotation_set_id != null && pendingImages.has(item.image_id))
      .map((item) => item.annotation_set_id as number))];
  }, [autoJobs, imageAnnotationStatuses]);

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
  const latestDataset = datasets[datasets.length - 1] ?? null;
  const selectedDataset = datasets.find((dataset) => dataset.version_id === selectedDatasetVersionId) ?? latestDataset;
  const latestModel = models[models.length - 1] ?? null;
  const selectedImageIndex = selectedImageId
    ? images.findIndex((image) => image.image_id === selectedImageId) + 1
    : 0;

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
    projectRequestId.current += 1;
    setClasses([]);
    setImages([]);
    setImageAnnotationStatuses([]);
    setAnnotations([]);
    setDatasets([]);
    setSelectedDatasetVersionId(null);
    setRuns([]);
    setModels([]);
    setAutoJobs([]);
    setDraftBoxes([]);
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
    setEditingClassId(null);
    setAnnotationImageId(null);
    setAutoLabelResults(null);
    setReviewToOpen(null);
    setTrainingSetupOpen(false);
    setSelectedAutoLabelModelId(null);
    setSelectedTestModelId(null);
    setTestImages([]);
    setTestPredictions([]);
    setTestConfidenceThreshold(0.25);
    setWorkspaceSection("annotate");
    setImageManagerOpen(false);
    setImageSelectorOpen(false);
    setPendingImportArchive(null);
    setArchiveInspection(null);
    setDatasetToView(null);
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
      setAnnotations(await api.listAnnotations(imageId));
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
    }).catch(() => {
      clearAuthSession();
      setAuthChecked(true);
      setSession(null);
    });
  }, [refreshAll, session?.access_token]);

  useEffect(() => {
    setDraftBoxes([]);
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
      return;
    }
    if (!selectedClassId || !classes.some((labelClass) => labelClass.class_id === selectedClassId)) {
      setSelectedClassId(classes[0].class_id);
    }
  }, [classes, selectedClassId]);

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

  useEffect(() => () => {
    testImages.forEach((image) => URL.revokeObjectURL(image.url));
  }, [testImages]);

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
    setDraftBoxes((current) => current.slice(0, -1));
  }, [poseCollecting]);

  const rememberReviewBoxes = useCallback(() => {
    setReviewHistory((current) => [...current, reviewBoxes.map(cloneReviewBox)]);
  }, [reviewBoxes]);

  const undoReviewEdit = useCallback(() => {
    setReviewHistory((current) => {
      const previous = current[current.length - 1];
      if (!previous) return current;
      setReviewBoxes(previous.map(cloneReviewBox));
      setSelectedReviewBoxIndex(previous.length > 0 ? 0 : null);
      setSelectedPoseKeypointIndex(0);
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
      if (target?.tagName === "INPUT" || target?.tagName === "SELECT" || target?.tagName === "TEXTAREA") return;
      if ((event.key === "Backspace" || event.key === "Delete") && editableSet && selectedReviewBoxIndex != null) {
        event.preventDefault();
        deleteSelectedReviewBox();
      } else if ((event.key === "Backspace" || event.key === "Delete") && draftBoxes.length > 0) {
        event.preventDefault();
        undoDraftBox();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteSelectedReviewBox, draftBoxes.length, editableSet, selectedReviewBoxIndex, undoDraftBox]);

  useEffect(() => {
    if (!autoLabelResults) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAutoLabelResults(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [autoLabelResults]);

  const runAction = async (label: string, action: () => Promise<unknown>, options: { refresh?: boolean } = {}) => {
    setBusy(true);
    setStatus(`${label}...`);
    try {
      await action();
      if (options.refresh !== false) {
        if (selectedProjectId) await refreshProject(selectedProjectId);
        if (selectedImageId) await refreshAnnotations(selectedImageId);
      }
      setStatus(`${label} done`);
    } catch (error) {
      setStatus(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const activateProject = async (projectId: number) => {
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

  const uploadImages = (fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0 || !selectedProjectId) return;
    void runAction(`Upload ${files.length} images`, async () => {
      let lastImage: ImageItem | null = null;
      for (const file of files) {
        lastImage = await api.uploadImage(selectedProjectId, file);
      }
      if (lastImage) setSelectedImageId(lastImage.image_id);
    });
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

  const importData = (fileList: FileList | File[] | null) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0 || !selectedProjectId) return;
    void runAction(`Import ${files.length} file${files.length === 1 ? "" : "s"}`, async () => {
      for (const file of files) {
        if (file.name.toLowerCase().endsWith(".zip")) {
          const inspection = await api.inspectArchive(file);
          if (inspection.archive_type === "YOLO") {
            setPendingImportArchive(file);
            setArchiveInspection(inspection);
            setImportLimit("");
            setImportClassMode("all");
            setImportClassIndexes("");
            setImportClassNames(inspection.class_names.join(", "));
            continue;
          }
          await api.importImageArchive(selectedProjectId, file);
          continue;
        }
        if (!file.type.startsWith("image/")) {
          throw new Error(`Unsupported file: ${file.name}`);
        }
        await api.uploadImage(selectedProjectId, file);
      }
    });
  };

  const confirmYoloImport = () => {
    const file = pendingImportArchive;
    if (!file || !selectedProjectId) return;
    const indexValue = importClassIndexes.trim();
    const importAllClasses = importClassMode === "all";
    if (!importAllClasses && !indexValue) {
      setStatus("Enter at least one YOLO class index");
      return;
    }
    const selectedIndexes = importAllClasses ? undefined : indexValue;
    const selectedClassNames = importClassNames.trim() || undefined;
    void runAction("Import YOLO dataset", async () => {
      const limit = Number(importLimit);
      const result = await api.importYoloZip(
        selectedProjectId,
        file,
        selectedClassNames,
        Number.isFinite(limit) && limit > 0 ? limit : null,
        selectedIndexes,
      );
      setStatus(
        `Imported ${result.imported_images} images, ${result.imported_annotation_sets} labels, ${result.imported_boxes} boxes`,
      );
      setPendingImportArchive(null);
      setArchiveInspection(null);
    });
  };

  const removeDraftBox = (index: number) => {
    setDraftBoxes((current) => current.filter((_, itemIndex) => itemIndex !== index));
  };

  const updateDraftBox = (index: number, box: DraftBox) => {
    setDraftBoxes((current) => current.map((item, itemIndex) => itemIndex === index ? box : item));
  };

  const startPoseCollection = () => {
    if (!selectedImageId || selectedClassId == null || editableSet) return;
    setPosePoints([]);
    setPoseVisibilityMode("visible");
    setPoseCollecting(true);
    setStatus("Mark each keypoint as visible, occluded, or outside the frame")
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
    const project = projects.find((item) => item.project_id === selectedProjectId);
    if (!window.confirm(`Delete project "${project?.name ?? selectedProjectId}" and all data inside it?`)) return;
    void runAction("Delete project", async () => {
      await api.deleteProject(selectedProjectId);
      const nextProjects = await api.listProjects();
      setProjects(nextProjects);
      setSelectedProjectId(nextProjects[0]?.project_id ?? null);
      clearProjectState();
      setActiveView("dashboard");
    }, { refresh: false });
  };

  const saveManualAnnotation = () => {
    if (!selectedImageId || draftBoxes.length === 0) return;
    const pendingIds = pendingSuggestions.map((set) => set.annotation_set_id);
    void runAction("Save annotation", async () => {
      await api.createAnnotationSet(selectedImageId, {
        annotations: draftBoxes.map((box) => ({ ...box, confidence: null })),
      });
      // A human label replaces the pending model suggestion for this image.
      if (pendingIds.length > 0) await api.rejectAnnotationsBulk(pendingIds);
      setDraftBoxes([]);
    });
  };

  const createAndBuildDataset = () => {
    if (!selectedProjectId) return;
    void runAction("Create and build dataset version", async () => {
      const dataset = await api.createDataset({
        project_id: selectedProjectId,
        preprocess_config: { resize: { width: 640, height: 640 } },
        augmentation_config: { horizontal_flip: true },
        default_split: "TRAIN",
        auto_split: true,
      });
      await api.buildYolo(dataset.version_id);
      setSelectedDatasetVersionId(dataset.version_id);
    });
  };

  const deleteDatasetVersion = (dataset: DatasetVersion) => {
    if (!window.confirm(`Delete dataset version ${dataset.version_number}? This cannot be undone.`)) return;
    void runAction("Delete dataset version", async () => {
      await api.deleteDataset(dataset.version_id);
      setDatasetToView(null);
    });
  };

  const trainYolo = () => {
    if (!selectedProjectId || !selectedDataset) return;
    setTrainingSetupOpen(false);
    void runAction("Train YOLO", async () => {
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
      await api.startYoloTraining(run.training_run_id);
    });
  };

  const chooseTestImages = (fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0) return;
    setTestImages(files.map((file, index) => ({
      key: `${file.name}-${file.size}-${file.lastModified}-${index}`,
      file,
      url: URL.createObjectURL(file),
    })));
    setTestPredictions([]);
  };

  const runModelTest = () => {
    if (!selectedTestModelId || testImages.length === 0 || !selectedProjectId) return;
    void runAction("Run model test", async () => {
      const batchSize = 100;
      const results: ModelPrediction[] = [];
      setTestPredictions([]);

      for (let offset = 0; offset < testImages.length; offset += batchSize) {
        const files = testImages.slice(offset, offset + batchSize).map((image) => image.file);
        setStatus(`Running test batch ${Math.floor(offset / batchSize) + 1}/${Math.ceil(testImages.length / batchSize)}`);
        const prediction = selectedTestModelId === "pose-pretrained"
          ? await api.predictPretrainedPoseBatch(selectedProjectId, files, testConfidenceThreshold)
          : await api.predictModelBatch(selectedTestModelId, files, testConfidenceThreshold);
        results.push(...prediction.results);
        setTestPredictions([...results]);
      }
    }, { refresh: false });
  };

  const showAutoLabelResults = async (job: AutoLabelJob) => {
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

  const autoLabel = () => {
    const selectedModel = typeof selectedAutoLabelModelId === "number"
      ? models.find((model) => model.model_id === selectedAutoLabelModelId)
      : null;
    if (!selectedProjectId || !selectedAutoLabelModelId || images.length === 0) return;
    void runAction("Auto label", async () => {
      const job = await api.createAutoLabelJob({
        project_id: selectedProjectId,
        model_id: selectedModel?.model_id ?? null,
        base_model: selectedAutoLabelModelId === "pose-pretrained" ? "yolo26n-pose.pt" : null,
        image_ids: images.map((image) => image.image_id),
        confidence_threshold: 0.25,
        max_predictions_per_image: 3,
        only_unlabeled: true,
      });
      await api.startYoloAutoLabelAsync(job.job_id);
      await waitForAutoLabelJob(job.job_id);
    });
  };

  const waitForAutoLabelJob = async (jobId: number) => {
    for (;;) {
      await sleep(1000);
      const job = await api.getAutoLabelJob(jobId);
      setStatus(`Auto label ${job.processed_images ?? 0}/${job.total_images ?? 0} ${job.status}`);
      if (selectedProjectId) setAutoJobs(await api.listAutoLabelJobs(selectedProjectId));
      if (job.status === "COMPLETED") {
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

  const startReviewEditor = (set: AnnotationSet) => {
    setDraftBoxes([]);
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
    setReviewSetId(null);
    setManualEditSetId(latestApproved.annotation_set_id);
    setReviewBoxes(latestApproved.annotations.map(toReviewBox));
    setReviewHistory([]);
    setSelectedReviewBoxIndex(latestApproved.annotations.length > 0 ? 0 : null);
    setSelectedPoseKeypointIndex(0);
    setPoseEditVisibilityMode("visible");
  };

  const openAutoLabelResult = (result: AutoLabelResult) => {
    setAutoLabelResults(null);
    setSelectedImageId(result.image.image_id);
    if (!result.annotationSet) return;
    setReviewToOpen(result.annotationSet);
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
    if (!window.confirm(`Accept all ${pendingIds.length} pending auto-label results into the next dataset version?`)) return;
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

  const review = (set: AnnotationSet, mode: "accept" | "reject") => {
    void runAction(`${mode} suggestion`, async () => {
      if (mode === "accept") await api.acceptAnnotation(set.annotation_set_id);
      if (mode === "reject") await api.rejectAnnotation(set.annotation_set_id);
      if (set.annotation_set_id === reviewSetId) closeReviewEditor();
    });
  };

  const logout = () => {
    const refreshToken = session?.refresh_token;
    clearAuthSession();
    setSession(null);
    clearProjectState();
    setProjects([]);
    if (refreshToken) void api.logout(refreshToken).catch(() => undefined);
  };

  const download = async (label: string, getBlob: () => Promise<Blob>, filename: string) => {
    try {
      setBusy(true);
      setStatus(`${label}...`);
      const url = URL.createObjectURL(await getBlob());
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`${label} done`);
    } catch (error) {
      setStatus(errorMessage(error));
    } finally {
      setBusy(false);
    }
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
              <div className={`workspace-nav-group ${["train", "models", "test"].includes(workspaceSection) ? "active" : ""}`}>
                <span><Layers3 size={17} /> Models</span>
                <button className={workspaceSection === "train" ? "active" : ""} onClick={() => setWorkspaceSection("train")} type="button"><Settings2 size={15} /> Train</button>
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
                  {images.slice(0, 8).map((image) => <button className="source-image-card" key={image.image_id} onClick={() => { setSelectedImageId(image.image_id); setWorkspaceSection("annotate"); }} type="button"><ProtectedImage image={image} alt={image.original_filename ?? `Image ${image.image_id}`} /><span>{image.original_filename ?? `Image ${image.image_id}`}</span></button>)}
                  {images.length === 0 && <label className="data-empty-drop"><Upload size={25} /><strong>Drop images or a ZIP here</strong><span>Image archives and YOLO datasets are detected automatically.</span><input type="file" accept="image/*,.zip,application/zip" multiple onChange={(event) => { importData(event.target.files); event.currentTarget.value = ""; }} /></label>}
                </div>
              </section>
              <section className="section-panel dataset-section">
                <div className="panel-heading"><strong>Version control</strong><span>View, download, or remove dataset snapshots.</span></div>
                <div className="dataset-version-list">
                  {[...datasets].reverse().map((dataset) => <div className="dataset-version-row" key={dataset.version_id}>
                    <div><strong>Version {dataset.version_number}</strong><span>{dataset.items.length} labeled images · {new Date(dataset.created_at).toLocaleDateString()}</span></div>
                    <button className="icon-button" onClick={() => setDatasetToView(dataset)} title="View version"><Eye size={16} /></button>
                    <button className="icon-button" onClick={() => void download("Download dataset", () => api.downloadDataset(dataset.version_id), `dataset-v${dataset.version_number}.zip`)} disabled={busy} title="Download version"><Download size={16} /></button>
                    <button className="icon-button danger-button" onClick={() => deleteDatasetVersion(dataset)} disabled={busy} title="Delete version"><Trash2 size={16} /></button>
                  </div>)}
                  {datasets.length === 0 && <div className="compact-empty">No dataset versions yet. Create one after labeling images in Annotate.</div>}
                </div>
              </section>
            </main>
          )}

          {workspaceSection === "annotate" && <main className="workspace annotation-workspace">
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
            {classes.map((labelClass) => editingClassId === labelClass.class_id ? (
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
              <div className={labelClass.class_id === selectedClassId ? "list-row active" : "list-row"} key={labelClass.class_id}>
                <button className="class-select" onClick={() => setSelectedClassId(labelClass.class_id)}>
                  <span className="swatch" style={{ background: labelClass.color }} />
                  <span>{labelClass.class_index}: {labelClass.name}</span>
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
            {poseProject && <button className="annotation-primary-action" onClick={startPoseCollection} disabled={!selectedImageId || selectedClassId == null || busy || !!editableSet || poseCollecting} title="Click 17 body keypoints">
              <Plus size={16} /> Add pose
            </button>}
            {latestApproved && <button onClick={startManualEditor} disabled={busy || !!editableSet} title="Edit saved labels">
              <Settings2 size={16} /> Edit labels
            </button>}
            {poseProject && (poseCollecting || draftBoxes.length > 0) && <div className="annotation-action-grid">
              <button className="primary" onClick={saveManualAnnotation} disabled={draftBoxes.length === 0 || busy || !!editableSet} title="Save labels">
                <Save size={16} /> Save
              </button>
              <button onClick={undoDraftBox} disabled={(poseCollecting ? posePoints.length === 0 : draftBoxes.length === 0) || busy || !!editableSet} title="Undo last label">
                <Undo2 size={16} /> Undo
              </button>
              <button onClick={() => poseCollecting ? setPosePoints([]) : setDraftBoxes([])} disabled={(poseCollecting ? posePoints.length === 0 : draftBoxes.length === 0) || busy || !!editableSet} title="Clear draft labels">
                <X size={16} /> Clear
              </button>
            </div>}
            {poseCollecting && <div className="pose-sidebar-progress">
              <div><strong>Adding pose</strong><span>{posePoints.length}/17 · {POSE_KEYPOINT_NAMES[posePoints.length] ?? "done"}</span></div>
              <button className={poseVisibilityMode === "visible" ? "active" : ""} onClick={() => setPoseVisibilityMode("visible")} type="button">Visible</button>
              <button className={poseVisibilityMode === "occluded" ? "active" : ""} onClick={() => setPoseVisibilityMode("occluded")} type="button">Occluded</button>
              <button className="outside-frame-action" onClick={skipPosePoint} type="button">Outside frame</button>
              <button onClick={() => { setPoseCollecting(false); setPosePoints([]); }} type="button">Cancel</button>
            </div>}
          </div>
        </aside>

        <section className="canvas-column">
          <AnnotationCanvas
            image={selectedImage}
            boxes={editableSet ? [] : latestApproved?.annotations ?? []}
            draftBoxes={draftBoxes}
            suggestions={editableSet ? [] : pendingSuggestions.flatMap((set) => set.annotations)}
            editableBoxes={editableSet ? reviewBoxes : []}
            selectedEditableIndex={selectedReviewBoxIndex}
            classId={selectedClassId}
            classColors={classColorById}
            onCreateBox={(box) => editableSet ? addReviewBox(box) : setDraftBoxes((current) => [...current, box])}
            onDraftChange={updateDraftBox}
            onDraftCancel={removeDraftBox}
            onSaveDraft={saveManualAnnotation}
            onEditableSelect={setSelectedReviewBoxIndex}
            onEditableChange={updateReviewBox}
            onEditableEditStart={rememberReviewBoxes}
            onUploadImages={uploadImages}
            poseCollecting={poseCollecting}
            posePoints={posePoints}
            onPosePoint={addPosePoint}
            poseEditing={poseProject && !!editableSet && selectedReviewBoxIndex != null}
            poseEditingKeypointIndex={selectedPoseKeypointIndex}
            onPoseEditPoint={(point) => updateSelectedPoseKeypoint(point, poseEditVisibilityMode === "visible" ? 2 : 1)}
            busy={busy}
          />
          {editableSet && (
            <div className="review-editor">
              <div className="review-editor-summary">
                <strong>{editingManualLabels ? `Edit revision ${editableSet.revision}` : `Review set ${editableSet.annotation_set_id}`}</strong>
                <span>{reviewBoxes.length} boxes</span>
              </div>
              <label className="review-class-field">
                Selected class
                <select
                  value={selectedReviewBoxIndex == null ? "" : reviewBoxes[selectedReviewBoxIndex]?.class_id ?? ""}
                  onChange={(event) => changeSelectedReviewClass(Number(event.target.value))}
                  disabled={selectedReviewBoxIndex == null}
                >
                  <option value="" disabled>Select a box</option>
                  {classes.map((labelClass) => (
                    <option key={labelClass.class_id} value={labelClass.class_id}>{labelClass.name}</option>
                  ))}
                </select>
              </label>
              {poseProject && reviewBoxes.length > 0 && (
                <div className="pose-editor-controls">
                  <label>
                    <span>Person</span>
                    <select
                      value={selectedReviewBoxIndex ?? ""}
                      onChange={(event) => {
                        setSelectedReviewBoxIndex(Number(event.target.value));
                        setSelectedPoseKeypointIndex(0);
                      }}
                    >
                      {reviewBoxes.map((_, index) => <option value={index} key={index}>Person {index + 1}</option>)}
                    </select>
                  </label>
                  <label>
                    <span>Keypoint</span>
                    <select value={selectedPoseKeypointIndex} onChange={(event) => setSelectedPoseKeypointIndex(Number(event.target.value))}>
                      {POSE_KEYPOINT_NAMES.map((name, index) => (
                        <option value={index} key={name}>{index + 1}. {name}</option>
                      ))}
                    </select>
                  </label>
                  <div className="pose-visibility-toggle" aria-label="Keypoint visibility">
                    <button className={poseEditVisibilityMode === "visible" ? "active" : ""} onClick={() => setPoseEditVisibilityMode("visible")} type="button">Visible</button>
                    <button className={poseEditVisibilityMode === "occluded" ? "active" : ""} onClick={() => setPoseEditVisibilityMode("occluded")} type="button">Occluded</button>
                    <button onClick={() => updateSelectedPoseKeypoint(null, 0)} type="button">Outside frame</button>
                  </div>
                  <span className="pose-editor-hint">Choose a state, then click the point on the image.</span>
                </div>
              )}
              <button className="danger-button" onClick={deleteSelectedReviewBox} disabled={selectedReviewBoxIndex == null || busy} title="Delete selected noise box">
                <Trash2 size={16} /> Delete box
              </button>
              <button onClick={undoReviewEdit} disabled={reviewHistory.length === 0 || busy} title="Undo review edit">
                <Undo2 size={16} /> Undo
              </button>
              <button onClick={resetReviewBoxes} disabled={busy} title="Restore labels before editing">
                <RefreshCw size={16} /> Reset
              </button>
              <button className="primary" onClick={saveEditedLabels} disabled={busy}>
                <Save size={16} /> {editingManualLabels ? "Save revision" : "Save corrected"}
              </button>
              <button onClick={closeReviewEditor} disabled={busy} title="Cancel label editing">
                <X size={16} /> Cancel
              </button>
            </div>
          )}
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
                {poseProject && <option value="pose-pretrained">YOLO26 Pose pretrained (label assist)</option>}
                {!poseProject && models.length === 0 && <option value="">No trained model</option>}
                {[...models].reverse().map((model) => (
                  <option value={model.model_id} key={model.model_id}>
                    Model v{model.version} · mAP50 {formatMetric(model.map50)}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={autoLabel} disabled={!selectedAutoLabelModelId || images.length === 0 || busy}><Play size={16} /> {poseProject ? "Generate pose suggestions" : "Run YOLO"}</button>
            <Metric label="Jobs" value={autoJobs.length} />
            <Metric label="Pending" value={pendingReviewImageIds.size} />
          </Panel>
          <Panel title="Dataset">
            <span className="panel-note">Save approved labels as a fixed training snapshot.</span>
            <button className="primary" onClick={createAndBuildDataset} disabled={!selectedProjectId || images.length === 0 || busy}><Database size={16} /> Create & Build version</button>
            <Metric label="Versions" value={datasets.length} />
            <Metric label="Latest" value={latestDataset ? `v${latestDataset.version_number}` : "-"} />
          </Panel>
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
              </div>
            </section>
          </main>
        )}

        {workspaceSection === "models" && (
          <main className="section-page">
            <PageHeading eyebrow="MODELS / DOWNLOAD" title="Download models" description="Choose a trained version and the format required by your deployment target." />
            <section className="model-version-list">
              {[...models].reverse().map((model) => (
                <article className="section-panel model-version-row" key={model.model_id}>
                  <span className="model-version-icon"><Layers3 size={20} /></span>
                  <div><strong>Model v{model.version}</strong><span>Trained artifact</span></div>
                  <Metric label="mAP50" value={formatMetric(model.map50)} />
                  <label className="model-format-field"><span>Format</span><select value={modelDownloadFormats[model.model_id] ?? "pt"} onChange={(event) => setModelDownloadFormats((current) => ({ ...current, [model.model_id]: event.target.value as ModelDownloadFormat }))}><option value="pt">PyTorch (.pt)</option><option value="onnx">ONNX (.onnx)</option><option value="torchscript">TorchScript (.torchscript)</option></select></label>
                  <button onClick={() => { const format = modelDownloadFormats[model.model_id] ?? "pt"; void download("Download model", () => api.downloadModel(model.model_id, format), `model-v${model.version}.${format === "torchscript" ? "torchscript" : format}`); }} disabled={busy}><Download size={16} /> Download</button>
                </article>
              ))}
              {models.length === 0 && <section className="section-panel section-empty"><Layers3 size={26} /><strong>No trained models yet</strong><span>Build a dataset and start a training run first.</span><button onClick={() => setWorkspaceSection("train")}><ChevronRight size={16} /> Go to training</button></section>}
            </section>
          </main>
        )}

        {workspaceSection === "test" && <ModelTestWorkspace
          models={models}
          projectId={selectedProjectId}
          poseProject={poseProject}
          selectedModelId={selectedTestModelId}
          images={testImages}
          predictions={testPredictions}
          confidenceThreshold={testConfidenceThreshold}
          classColors={classColorById}
          busy={busy}
          onModelChange={setSelectedTestModelId}
          onImagesChange={chooseTestImages}
          onThresholdChange={setTestConfidenceThreshold}
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

      {autoLabelResults && (
        <AutoLabelResultsModal
          results={autoLabelResults}
          classColors={classColorById}
          onClose={() => setAutoLabelResults(null)}
          onOpen={openAutoLabelResult}
          onAcceptAll={acceptAllAutoLabelResults}
          onRejectAll={rejectAllAutoLabelResults}
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
          pendingImageIds={pendingReviewImageIds}
          annotationStatuses={imageAnnotationStatuses}
          pendingAnnotationSetCount={pendingProjectAnnotationSetIds.length}
          busy={busy}
          onSelect={(imageId) => { setSelectedImageId(imageId); setImageSelectorOpen(false); }}
          onAcceptAll={() => reviewAllProjectSuggestions("accept")}
          onRejectAll={() => reviewAllProjectSuggestions("reject")}
          onClose={() => setImageSelectorOpen(false)}
        />
      )}

      {pendingImportArchive && archiveInspection && (
        <YoloImportModal
          file={pendingImportArchive}
          inspection={archiveInspection}
          limit={importLimit}
          classMode={importClassMode}
          classIndexes={importClassIndexes}
          classNames={importClassNames}
          busy={busy}
          onLimitChange={setImportLimit}
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

      {activeView === "workspace" && workspaceSection === "annotate" && (
        <section className="review-band">
          <SectionTitle title="Review Queue" />
          {pendingSuggestions.length === 0 ? <span className="muted">No pending model suggestions</span> : pendingSuggestions.map((set) => (
            <div className="review-row" key={set.annotation_set_id}>
              <span>Set {set.annotation_set_id} · revision {set.revision} · {set.annotations.length} boxes</span>
              <div>
                <button onClick={() => review(set, "accept")} disabled={busy}><Check size={16} /> Accept all</button>
                <button onClick={() => startReviewEditor(set)} disabled={busy || reviewSetId === set.annotation_set_id}><Save size={16} /> Edit</button>
                <button onClick={() => review(set, "reject")} disabled={busy}><X size={16} /> Reject all</button>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function ImageSelectorModal({
  images,
  selectedImageId,
  classColors,
  pendingImageIds,
  annotationStatuses,
  pendingAnnotationSetCount,
  busy,
  onSelect,
  onAcceptAll,
  onRejectAll,
  onClose,
}: {
  images: ImageItem[];
  selectedImageId: number | null;
  classColors: Record<number, string>;
  pendingImageIds: Set<number>;
  annotationStatuses: ImageAnnotationStatus[];
  pendingAnnotationSetCount: number;
  busy: boolean;
  onSelect: (imageId: number) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  onClose: () => void;
}) {
  const batchSize = 24;
  const selectedIndex = Math.max(0, images.findIndex((image) => image.image_id === selectedImageId));
  const initialStart = Math.floor(selectedIndex / batchSize) * batchSize;
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "labeled" | "needs-drawing" | "approved" | "pending" | "rejected">("all");
  const [rangeStart, setRangeStart] = useState(initialStart);
  const [rangeEnd, setRangeEnd] = useState(Math.min(images.length, initialStart + batchSize));
  const [previews, setPreviews] = useState<Record<number, AnnotationSet | null>>({});
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
  const searched = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query ? images.filter((image, index) => `${index + 1} ${image.original_filename ?? ""}`.toLowerCase().includes(query)) : images;
  }, [images, search]);
  const filtered = useMemo(() => searched.filter((image) => {
    const status = statusByImage.get(image.image_id);
    if (filter === "labeled") return !!status && status.box_count > 0 && status.status !== "REJECTED";
    if (filter === "approved") return status?.status === "APPROVED";
    if (filter === "pending") return status?.status === "PENDING";
    if (filter === "rejected") return status?.status === "REJECTED";
    if (filter === "needs-drawing") return !status || status.box_count === 0;
    return true;
  }), [filter, searched, statusByImage]);
  const filterCounts = useMemo(() => ({
    all: images.length,
    labeled: images.filter((image) => { const status = statusByImage.get(image.image_id); return !!status && status.box_count > 0 && status.status !== "REJECTED"; }).length,
    approved: images.filter((image) => statusByImage.get(image.image_id)?.status === "APPROVED").length,
    pending: images.filter((image) => statusByImage.get(image.image_id)?.status === "PENDING").length,
    rejected: images.filter((image) => statusByImage.get(image.image_id)?.status === "REJECTED").length,
    "needs-drawing": images.filter((image) => {
      const status = statusByImage.get(image.image_id);
      return !status || status.box_count === 0;
    }).length,
  }), [images, statusByImage]);
  const visible = filtered.slice(rangeStart, rangeEnd);

  useEffect(() => {
    let active = true;
    const missing = visible.filter((image) => {
      if (!Object.prototype.hasOwnProperty.call(previews, image.image_id)) return true;
      return previews[image.image_id]?.status !== statusByImage.get(image.image_id)?.status;
    });
    if (missing.length === 0) return () => { active = false; };
    void Promise.all(missing.map(async (image) => {
      const sets = await api.listAnnotations(image.image_id);
      const preview = [...sets].reverse()[0] ?? null;
      return [image.image_id, preview] as const;
    })).then((entries) => {
      if (active) setPreviews((current) => ({ ...current, ...Object.fromEntries(entries) }));
    }).catch(() => undefined);
    return () => { active = false; };
  }, [rangeStart, rangeEnd, search, annotationStatuses]);

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
  }, [selectedImageId]);

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

  const updateSearch = (value: string) => {
    setSearch(value);
    setRangeStart(0);
    setRangeEnd(Math.min(batchSize, images.length));
  };

  const updateFilter = (value: typeof filter) => {
    setFilter(value);
    setRangeStart(0);
    setRangeEnd(Math.min(batchSize, images.length));
  };

  return <div className="results-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="image-selector-modal" role="dialog" aria-modal="true" aria-label="Select image">
      <header className="image-manager-header"><div><h3>Select image</h3><p>Labels and model suggestions are shown on each preview.</p></div><button className="icon-button" onClick={onClose} title="Close"><X size={18} /></button></header>
      <div className="image-selector-toolbar"><input value={search} onChange={(event) => updateSearch(event.target.value)} placeholder="Search filename or image number" /><span>{filtered.length} images</span></div>
      <div className="image-selector-filters" aria-label="Filter images">
        {([
          ["all", "All"], ["labeled", "Has labels"], ["needs-drawing", "Needs drawing"],
          ["approved", "Accepted"],
          ["pending", "Pending"], ["rejected", "Denied"],
        ] as const).map(([value, label]) => <button className={filter === value ? "active" : ""} onClick={() => updateFilter(value)} type="button" key={value}><span>{label}</span><strong>{filterCounts[value]}</strong></button>)}
      </div>
      <div className="image-selector-grid" ref={gridRef}>
        {rangeStart > 0 && <div className="image-lazy-sentinel" ref={topSentinelRef}>Loading earlier images...</div>}
        {visible.map((image) => {
          const globalIndex = images.findIndex((item) => item.image_id === image.image_id) + 1;
          const preview = previews[image.image_id];
          const annotationStatus = statusByImage.get(image.image_id);
          const displayStatus = annotationStatus?.status ?? "UNLABELED";
          const hasLabels = (annotationStatus?.box_count ?? 0) > 0;
          const statusText = displayStatus === "REJECTED"
            ? "Denied"
            : !hasLabels
              ? "Needs drawing"
              : displayStatus === "APPROVED"
                ? `Accepted · ${annotationStatus?.box_count ?? 0}`
                : `Pending · ${annotationStatus?.box_count ?? 0}`;
          const statusClass = displayStatus === "REJECTED" ? "rejected" : hasLabels ? displayStatus.toLowerCase() : "needs-drawing";
          const frame = getImageFrameForDimensions(image.width ?? 1, image.height ?? 1, 240 / 150);
          const selected = image.image_id === selectedImageId;
          return <button ref={selected ? selectedRef : undefined} className={`image-selector-card ${selected ? "active" : ""} ${hasLabels ? "has-labels" : "needs-drawing"}`} aria-pressed={selected} onClick={() => onSelect(image.image_id)} key={image.image_id} type="button">
            {selected && <span className="image-card-selected"><Check size={14} /> Selected image</span>}
            <span className={`image-card-label-state ${hasLabels ? "labeled" : "drawing"}`}>{hasLabels ? "Labeled" : "Needs drawing"}</span>
            <ImageSelectorPreview image={image} alt={image.original_filename ?? `Image ${globalIndex}`} preview={preview} frame={frame} classColors={classColors} />
            <div><strong>#{globalIndex} {image.original_filename ?? `image-${globalIndex}`}</strong><span className={statusClass}>{pendingImageIds.has(image.image_id) && hasLabels ? `Pending · ${annotationStatus?.box_count ?? 0}` : statusText}</span></div>
          </button>;
        })}
        <div className="image-lazy-sentinel" ref={bottomSentinelRef}>{rangeEnd < filtered.length ? "Loading more images..." : "All images loaded"}</div>
      </div>
      <footer className="image-selector-footer">
        <span>Showing {visible.length} of {filtered.length} images · scroll to load more</span>
        {pendingAnnotationSetCount > 0 && <div className="image-selector-review-actions">
          <button onClick={onAcceptAll} disabled={busy}><Check size={16} /> Accept all ({pendingAnnotationSetCount})</button>
          <button className="danger-button" onClick={onRejectAll} disabled={busy}><Trash2 size={16} /> Reject all ({pendingAnnotationSetCount})</button>
        </div>}
      </footer>
    </section>
  </div>;
}

function YoloImportModal({
  file, inspection, limit, classMode, classIndexes, classNames, busy,
  onLimitChange, onClassModeChange, onClassIndexesChange, onClassNamesChange, onImport, onClose,
}: {
  file: File;
  inspection: ArchiveInspection;
  limit: string;
  classMode: "all" | "selected";
  classIndexes: string;
  classNames: string;
  busy: boolean;
  onLimitChange: (value: string) => void;
  onClassModeChange: (value: "all" | "selected") => void;
  onClassIndexesChange: (value: string) => void;
  onClassNamesChange: (value: string) => void;
  onImport: () => void;
  onClose: () => void;
}) {
  return <div className="results-modal-backdrop"><section className="training-modal" role="dialog" aria-modal="true" aria-label="Import YOLO dataset">
    <header className="training-modal-header"><div><h3>YOLO dataset detected</h3><p>{file.name} · {inspection.image_count} images</p></div><button className="icon-button" onClick={onClose} disabled={busy}><X size={18} /></button></header>
    <div className="training-form">
      <label className="training-field"><span>Import images</span><input min="1" max={inspection.image_count} placeholder={`All ${inspection.image_count} images`} type="number" value={limit} onChange={(event) => onLimitChange(event.target.value)} /></label>
      <label className="training-field"><span>YOLO classes</span><select value={classMode} onChange={(event) => onClassModeChange(event.target.value as "all" | "selected")}><option value="all">All classes</option><option value="selected">Selected indexes</option></select></label>
      {classMode === "selected" && <label className="training-field"><span>Class indexes</span><input placeholder="0 or 0,1" value={classIndexes} onChange={(event) => onClassIndexesChange(event.target.value)} /></label>}
      <label className="training-field training-field-wide"><span>Class names</span><input value={classNames} onChange={(event) => onClassNamesChange(event.target.value)} placeholder="Detected from data.yaml" /></label>
    </div>
    <footer className="training-modal-footer"><button onClick={onClose} disabled={busy}>Cancel</button><button className="primary" onClick={onImport} disabled={busy}><Upload size={16} /> Import dataset</button></footer>
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
                      Model v{model.version} · mAP50 {formatMetric(model.map50)}
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

function useProtectedImageUrl(image: ImageItem | null) {
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
    void api.getImageBlob(image).then((blob) => {
      objectUrl = URL.createObjectURL(blob);
      if (active) setUrl(objectUrl);
      else URL.revokeObjectURL(objectUrl);
    }).catch(() => active && setUrl(null));
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageId, version]);

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
}: {
  image: ImageItem;
  alt: string;
  preview: AnnotationSet | null | undefined;
  frame: ImageFrame;
  classColors: Record<number, string>;
}) {
  const imageUrl = useProtectedImageUrl(image);
  const [imageReady, setImageReady] = useState(false);

  useEffect(() => setImageReady(false), [imageUrl, image.image_id]);

  const tone = preview?.status === "PENDING"
    ? "suggestion"
    : preview?.status === "REJECTED"
      ? "rejected"
      : "approved";

  return <div className="image-selector-preview">
    {imageUrl
      ? <img className={imageReady ? undefined : "image-hidden"} src={imageUrl} alt={alt} loading="lazy" onLoad={() => setImageReady(true)} />
      : <span className="protected-image-placeholder">Loading image...</span>}
    {imageReady && preview?.annotations.map((box, index) => <span key={index}>
      <Box box={box} frame={frame} tone={tone} color={classColors[box.class_id]} />
      <PoseKeypoints keypoints={(box.keypoints ?? []).map((point) => ({ x: point.x, y: point.y, visibility: point.visibility }))} frame={frame} tone={tone} />
    </span>)}
  </div>;
}

function ModelTestWorkspace({
  models,
  projectId,
  poseProject,
  selectedModelId,
  images,
  predictions,
  confidenceThreshold,
  classColors,
  busy,
  onModelChange,
  onImagesChange,
  onThresholdChange,
  onRun,
}: {
  models: ModelItem[];
  projectId: number | null;
  poseProject: boolean;
  selectedModelId: number | "pose-pretrained" | null;
  images: TestImage[];
  predictions: ModelPrediction[];
  confidenceThreshold: number;
  classColors: Record<number, string>;
  busy: boolean;
  onModelChange: (modelId: number | "pose-pretrained" | null) => void;
  onImagesChange: (files: FileList | null) => void;
  onThresholdChange: (value: number) => void;
  onRun: () => void;
}) {
  const [testPage, setTestPage] = useState(0);
  const [isFileDragActive, setIsFileDragActive] = useState(false);
  const pageSize = 48;
  const projectModels = useMemo(
    () => models.filter((model) => model.project_id === projectId),
    [models, projectId],
  );
  const selectedModel = typeof selectedModelId === "number"
    ? projectModels.find((model) => model.model_id === selectedModelId) ?? null
    : null;
  const detectedCount = predictions.reduce((total, prediction) => total + prediction.boxes.length, 0);
  const pageCount = Math.max(1, Math.ceil(images.length / pageSize));
  const currentPage = Math.min(testPage, pageCount - 1);
  const pageStart = currentPage * pageSize;
  const visibleImages = images.slice(pageStart, pageStart + pageSize);
  const canHandleFiles = (event: DragEvent<HTMLElement>) => Array.from(event.dataTransfer.types).includes("Files");

  useEffect(() => setTestPage(0), [images]);
  useEffect(() => {
    const invalidCustomModel = typeof selectedModelId === "number" && selectedModel == null;
    const invalidPoseModel = selectedModelId === "pose-pretrained" && !poseProject;
    if (invalidCustomModel || invalidPoseModel) onModelChange(null);
  }, [onModelChange, poseProject, selectedModel, selectedModelId]);

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
        <div className="test-toolbar">
          <label className="file-button">
            <Upload size={16} /> Choose test images
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(event) => {
                onImagesChange(event.target.files);
                event.currentTarget.value = "";
              }}
            />
          </label>
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
              {poseProject && <option value="pose-pretrained">YOLO26 Pose pretrained (base)</option>}
              {[...projectModels].reverse().map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  Model v{model.version} · mAP50 {formatMetric(model.map50)}
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
          <button className="primary" onClick={onRun} disabled={!selectedModelId || images.length === 0 || busy}>
            <Play size={16} /> Run {images.length || ""} test{images.length === 1 ? "" : "s"}
          </button>
        </div>
        {images.length === 0 ? <div className="test-image-canvas"><div className="empty">Choose test images</div></div>
          : <>
            <div className="test-results-scroll">
              <div className="test-results-grid">
                {visibleImages.map((image, index) => {
                  const predictionIndex = pageStart + index;
                  return <TestImageCard
                    key={image.key}
                    image={image}
                    prediction={predictions[predictionIndex] ?? null}
                    classColors={classColors}
                  />;
                })}
              </div>
            </div>
            <nav className="test-pager" aria-label="Test image pages">
              <span>Showing {pageStart + 1}-{Math.min(pageStart + pageSize, images.length)} of {images.length}</span>
              <div>
                <button
                  className="icon-button"
                  title="Previous page"
                  aria-label="Previous page"
                  onClick={() => setTestPage((page) => Math.max(0, page - 1))}
                  disabled={currentPage === 0}
                ><ChevronLeft size={18} /></button>
                <strong>{currentPage + 1} / {pageCount}</strong>
                <button
                  className="icon-button"
                  title="Next page"
                  aria-label="Next page"
                  onClick={() => setTestPage((page) => Math.min(pageCount - 1, page + 1))}
                  disabled={currentPage === pageCount - 1}
                ><ChevronRight size={18} /></button>
              </div>
            </nav>
          </>}
        {isFileDragActive && <div className="test-drop-overlay"><Upload size={28} /><strong>Drop test images here</strong></div>}
      </section>

      <aside className="rightbar test-sidebar">
        <Panel title="Test run">
          <Metric label="Model" value={selectedModelId === "pose-pretrained" ? "Pose base" : selectedModel ? `v${selectedModel.version}` : "-"} />
          <Metric label="Threshold" value={confidenceThreshold.toFixed(2)} />
          <Metric label="Images" value={images.length} />
          <Metric label="Detections" value={predictions.length > 0 ? detectedCount : "-"} />
        </Panel>
        <Panel title="Predictions">
          {predictions.length === 0 ? <span className="muted">Choose test images and run the selected model.</span>
            : detectedCount === 0 ? <span className="muted">No detections above the selected threshold.</span>
              : <div className="test-prediction-list">
                {predictions.flatMap((prediction, imageIndex) => prediction.boxes.map((box, boxIndex) => (
                  <div className="test-prediction-row" key={`${imageIndex}-${box.class_id}-${boxIndex}`}>
                    <span className="swatch" style={{ background: classColors[box.class_id] ?? "#16a34a" }} />
                    <strong>{images[imageIndex]?.file.name}: {box.class_name}</strong>
                    <span>{box.confidence.toFixed(2)}</span>
                  </div>
                ))) }
              </div>}
        </Panel>
      </aside>
    </main>
  );
}

function TestImageCard({
  image,
  prediction,
  classColors,
}: {
  image: TestImage;
  prediction: ModelPrediction | null;
  classColors: Record<number, string>;
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
        <span>{prediction ? `${prediction.boxes.length} detected` : "Ready"}</span>
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

function AutoLabelResultsModal({
  results,
  classColors,
  onClose,
  onOpen,
  onAcceptAll,
  onRejectAll,
  busy,
}: {
  results: AutoLabelResult[];
  classColors: Record<number, string>;
  onClose: () => void;
  onOpen: (result: AutoLabelResult) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  busy: boolean;
}) {
  const [resultFilter, setResultFilter] = useState<AutoLabelResultFilter>("all");
  const [resultsPage, setResultsPage] = useState(0);
  const resultsGridRef = useRef<HTMLDivElement | null>(null);
  const resultsPageSize = 60;
  const isAccepted = (result: AutoLabelResult) => result.annotationSet?.status === "APPROVED";
  const activeResults = results.filter((result) => !isAccepted(result));
  const detectedCount = activeResults.filter((result) => (result.annotationSet?.annotations.length ?? 0) > 0).length;
  const emptyCount = results.filter(
    (result) => !isAccepted(result) && result.status === "COMPLETED" && (result.annotationSet?.annotations.length ?? 0) === 0,
  ).length;
  const failedCount = activeResults.filter((result) => result.status === "FAILED").length;
  const acceptedCount = results.filter(isAccepted).length;
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
  const resultsPageCount = Math.max(1, Math.ceil(filteredResults.length / resultsPageSize));
  const visibleResults = filteredResults.slice(
    resultsPage * resultsPageSize,
    (resultsPage + 1) * resultsPageSize,
  );
  const filters: { key: AutoLabelResultFilter; label: string; count: number }[] = [
    { key: "all", label: "All", count: activeResults.length },
    { key: "detected", label: "Detected", count: detectedCount },
    { key: "empty", label: "No boxes", count: emptyCount },
    { key: "failed", label: "Failed", count: failedCount },
    { key: "accepted", label: "Accepted", count: acceptedCount },
  ];

  useEffect(() => {
    setResultsPage(0);
    resultsGridRef.current?.scrollTo({ top: 0 });
  }, [resultFilter]);

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
            <p>{results.length} images processed</p>
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
                    <ProtectedImage image={result.image} alt={result.image.original_filename ?? `Image ${result.image.image_id}`} />
                    {result.annotationSet?.annotations.map((box, index) => (
                      <span
                        className="result-box"
                        key={`${result.annotationSet?.annotation_set_id}-${box.annotation_id ?? index}`}
                        style={{
                          borderColor: classColors[box.class_id] ?? "#f59e0b",
                          left: `${box.x_min * 100}%`,
                          top: `${box.y_min * 100}%`,
                          width: `${(box.x_max - box.x_min) * 100}%`,
                          height: `${(box.y_max - box.y_min) * 100}%`,
                        }}
                      />
                    ))}
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
          </div>
        )}
        {filteredResults.length > resultsPageSize && (
          <nav className="results-pager" aria-label="Auto-label result pages">
            <span>
              Showing {resultsPage * resultsPageSize + 1}-{Math.min((resultsPage + 1) * resultsPageSize, filteredResults.length)} of {filteredResults.length}
            </span>
            <div>
              <button
                className="icon-button"
                onClick={() => {
                  setResultsPage((page) => Math.max(0, page - 1));
                  resultsGridRef.current?.scrollTo({ top: 0 });
                }}
                disabled={resultsPage === 0 || busy}
                title="Previous results page"
              ><ChevronLeft size={16} /></button>
              <strong>{resultsPage + 1} / {resultsPageCount}</strong>
              <button
                className="icon-button"
                onClick={() => {
                  setResultsPage((page) => Math.min(resultsPageCount - 1, page + 1));
                  resultsGridRef.current?.scrollTo({ top: 0 });
                }}
                disabled={resultsPage >= resultsPageCount - 1 || busy}
                title="Next results page"
              ><ChevronRight size={16} /></button>
            </div>
          </nav>
        )}
        <footer className="results-modal-footer">
          <span>Select an image to inspect, edit, accept, or reject its predictions.</span>
          <div className="results-modal-actions">
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

function AnnotationCanvas({
  image,
  boxes,
  draftBoxes,
  suggestions,
  editableBoxes,
  selectedEditableIndex,
  classId,
  classColors,
  onCreateBox,
  onDraftChange,
  onDraftCancel,
  onSaveDraft,
  onEditableSelect,
  onEditableChange,
  onEditableEditStart,
  onUploadImages,
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
  classColors: Record<number, string>;
  onCreateBox: (box: DraftBox) => void;
  onDraftChange: (index: number, box: DraftBox) => void;
  onDraftCancel: (index: number) => void;
  onSaveDraft: () => void;
  onEditableSelect: (index: number | null) => void;
  onEditableChange: (index: number, box: ReviewBox) => void;
  onEditableEditStart: () => void;
  onUploadImages: (files: FileList | null) => void;
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
  const [isFileDragActive, setIsFileDragActive] = useState(false);
  const [draftMenu, setDraftMenu] = useState<{ index: number; left: number; top: number; editing: boolean } | null>(null);
  const [editDrag, setEditDrag] = useState<{
    index: number;
    mode: EditMode;
    start: { x: number; y: number };
    box: ReviewBox;
  } | null>(null);
  const imageId = image?.image_id ?? null;
  const imageUrl = useProtectedImageUrl(image);
  const imageFrame = useMemo(() => getImageFrame(image), [image]);

  useEffect(() => {
    setImageReady(false);
    setStart(null);
    setCurrent(null);
    setEditDrag(null);
    setDraftMenu(null);
  }, [imageId]);

  useEffect(() => {
    if (draftMenu && draftMenu.index >= draftBoxes.length) setDraftMenu(null);
  }, [draftBoxes.length, draftMenu]);

  const draft = useMemo(() => {
    if (!start || !current || classId == null) return null;
    return normalizeBox(start, current, classId);
  }, [classId, current, start]);

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
      className={isFileDragActive ? "annotation-canvas file-drag-active" : "annotation-canvas"}
      ref={ref}
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
        onUploadImages(event.dataTransfer.files);
      }}
      onPointerDown={(event) => {
        if (!imageReady) {
          if (!image) fileInputRef.current?.click();
          return;
        }
        if (poseCollecting || poseEditing) return;
        onEditableSelect(null);
        setStart(point(event));
      }}
      onPointerMove={(event) => start && setCurrent(point(event))}
      onPointerUp={(event) => {
        if (poseCollecting && imageReady) {
          onPosePoint(point(event));
          return;
        }
        if (poseEditing && imageReady) {
          onPoseEditPoint(point(event));
          return;
        }
        if (!start || classId == null) return;
        const box = normalizeBox(start, point(event), classId);
        if (box.x_max - box.x_min > 0.01 && box.y_max - box.y_min > 0.01) {
          const rect = ref.current?.getBoundingClientRect();
          onCreateBox(box);
          if (rect) {
            setDraftMenu({
              index: draftBoxes.length,
              left: clampRange(event.clientX - rect.left, 12, Math.max(12, rect.width - 190)),
              top: clampRange(event.clientY - rect.top + 10, 12, Math.max(12, rect.height - 58)),
              editing: false,
            });
          }
        }
        setStart(null);
        setCurrent(null);
      }}
    >
      <input
        ref={fileInputRef}
        className="canvas-file-input"
        type="file"
        accept="image/*"
        multiple
        onChange={(event) => {
          onUploadImages(event.target.files);
          event.currentTarget.value = "";
        }}
      />
      {image ? (
        <>
          {imageUrl ? <img
            className={imageReady ? "" : "image-hidden"}
            src={imageUrl}
            alt={image.original_filename ?? "uploaded"}
            onLoad={() => setImageReady(true)}
            onError={() => setImageReady(false)}
          /> : <div className="empty">Loading image...</div>}
          {!imageReady && <div className="empty">Image file missing. Upload this image again.</div>}
        </>
      ) : (
        <div className="empty canvas-upload-empty"><Upload size={24} /><span>Drop images here or click to browse</span></div>
      )}
      {isFileDragActive && <div className="canvas-drop-overlay"><Upload size={28} /><strong>Drop images to upload</strong></div>}
      {imageReady && boxes.map((box, index) => <Box key={`box-${index}`} box={box} frame={imageFrame} tone="approved" color={classColors[box.class_id]} />)}
      {imageReady && suggestions.map((box, index) => <Box key={`suggestion-${index}`} box={box} frame={imageFrame} tone="suggestion" color={classColors[box.class_id]} />)}
      {imageReady && draftBoxes.map((box, index) => draftMenu?.editing && draftMenu.index === index ? (
        <EditableBox
          key={`draft-editable-${index}`}
          box={{ ...box, confidence: null }}
          frame={imageFrame}
          color={classColors[box.class_id]}
          selected
          onPointerDown={(event, mode) => {
            event.stopPropagation();
            event.currentTarget.setPointerCapture(event.pointerId);
            setEditDrag({ index, mode, start: point(event), box: { ...box, confidence: null } });
          }}
          onPointerMove={(event) => {
            if (!editDrag) return;
            event.stopPropagation();
            const nextPoint = point(event);
            const next = transformReviewBox(
              { ...editDrag.box, confidence: null },
              editDrag.mode,
              nextPoint.x - editDrag.start.x,
              nextPoint.y - editDrag.start.y,
            );
            const { confidence: _confidence, ...draftBox } = next;
            onDraftChange(index, draftBox);
          }}
          onPointerUp={(event) => {
            event.stopPropagation();
            setEditDrag(null);
          }}
        />
      ) : <Box key={`draft-${index}`} box={box} frame={imageFrame} tone="draft" color={classColors[box.class_id]} />)}
      {imageReady && boxes.map((box, index) => <PoseKeypoints key={`saved-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} />)}
      {imageReady && suggestions.map((box, index) => <PoseKeypoints key={`suggestion-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} tone="suggestion" />)}
      {imageReady && draftBoxes.map((box, index) => <PoseKeypoints key={`draft-pose-${index}`} keypoints={box.keypoints ?? []} frame={imageFrame} />)}
      {imageReady && poseCollecting && <PoseKeypoints keypoints={posePoints} frame={imageFrame} />}
      {imageReady && editableBoxes.map((box, index) => poseEditing ? (
        <span key={`editable-pose-${index}`}>
          <Box box={box} frame={imageFrame} tone={selectedEditableIndex === index ? "pose-editing selected" : "pose-editing"} color={classColors[box.class_id]} />
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
          selected={selectedEditableIndex === index}
          onPointerDown={(event, mode) => startEditableDrag(event, index, mode)}
          onPointerMove={moveEditableBox}
          onPointerUp={stopEditableDrag}
        />
      ))}
      {imageReady && draft && <Box box={draft} frame={imageFrame} tone="draft" color={classColors[draft.class_id]} />}
      {draftMenu && <div className="draft-context-menu" style={{ left: draftMenu.left, top: draftMenu.top }} onPointerDown={(event) => event.stopPropagation()}>
        <button className="primary" onClick={onSaveDraft} disabled={busy}><Save size={15} /> Save</button>
        <button onClick={() => setDraftMenu((current) => current ? { ...current, editing: true } : null)} disabled={busy || draftMenu.editing}><Settings2 size={15} /> Edit</button>
        <button className="icon-button" onClick={() => { onDraftCancel(draftMenu.index); setDraftMenu(null); }} disabled={busy} title="Cancel this box"><X size={15} /></button>
      </div>}
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
  selected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: {
  box: ReviewBox;
  frame: ImageFrame;
  color?: string;
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
      {box.confidence != null && <span>{box.confidence.toFixed(2)}</span>}
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
}: {
  box: Pick<AnnotationBox, "class_id" | "x_min" | "y_min" | "x_max" | "y_max"> & { confidence?: number | null };
  frame: ImageFrame;
  tone: string;
  color?: string;
}) {
  return (
    <div
      className={`box ${tone}`}
      style={boxStyle(box, frame, color)}
    >
      {box.confidence != null && <span>{box.confidence.toFixed(2)}</span>}
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

function normalizeBox(a: { x: number; y: number }, b: { x: number; y: number }, classId: number): DraftBox {
  return {
    class_id: classId,
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

function formatBaseModel(baseModel: string, models: ModelItem[]) {
  if (!baseModel.startsWith("model:")) return baseModel.replace(".pt", "");
  const modelId = Number(baseModel.slice("model:".length));
  const model = models.find((item) => item.model_id === modelId);
  return model ? `Model v${model.version}` : "Existing model";
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
