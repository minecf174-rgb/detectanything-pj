export type ShortcutContext = "detect" | "pose" | "pose-drawing" | "detect-edit" | "pose-edit" | "negative-review";

export type ShortcutDefinition = {
  id: string;
  group: string;
  label: string;
  description: string;
  defaultKey: string;
  contexts: ShortcutContext[];
};

const idle: ShortcutContext[] = ["detect", "pose"];
const editing: ShortcutContext[] = ["detect-edit", "pose-edit"];

export const SHORTCUT_DEFINITIONS: ShortcutDefinition[] = [
  { id: "previousImage", group: "Navigation", label: "Previous image", description: "Move to the image before this one when no edit is open, including no-box review.", defaultKey: "ArrowLeft", contexts: [...idle, "negative-review"] },
  { id: "nextImage", group: "Navigation", label: "Next image", description: "Move to the next image when no edit is open, including no-box review.", defaultKey: "ArrowRight", contexts: [...idle, "negative-review"] },
  { id: "save", group: "Editing", label: "Save labels", description: "Save a completed draft, or save the current revision/correction when editing.", defaultKey: "Mod+S", contexts: ["detect", "pose", "pose-drawing", ...editing] },
  { id: "cancelEdit", group: "Editing", label: "Cancel editing", description: "Close the revision/correction editor or cancel a pose being placed. Unsaved edits are discarded.", defaultKey: "Escape", contexts: ["pose-drawing", ...editing] },
  { id: "undoDraft", group: "Editing", label: "Undo draft", description: "Remove the last unsaved box or pose point.", defaultKey: "Backspace", contexts: ["detect", "pose", "pose-drawing"] },
  { id: "undoDraftAlt", group: "Editing", label: "Undo draft (alternate)", description: "Second key for removing the last unsaved box or pose point.", defaultKey: "Delete", contexts: ["detect", "pose", "pose-drawing"] },
  { id: "deleteBox", group: "Editing", label: "Delete selected box", description: "Remove the selected box while editing a revision/correction.", defaultKey: "Backspace", contexts: editing },
  { id: "deleteBoxAlt", group: "Editing", label: "Delete selected box (alternate)", description: "Second key for deleting the selected box.", defaultKey: "Delete", contexts: editing },
  { id: "acceptSuggestion", group: "Auto-label review", label: "Accept suggestion", description: "Accept the only pending suggestion on this image, when no draft is open.", defaultKey: "Y", contexts: idle },
  { id: "rejectSuggestion", group: "Auto-label review", label: "Reject suggestion", description: "Reject the only pending suggestion on this image, when no draft is open.", defaultKey: "N", contexts: idle },
  { id: "editSuggestion", group: "Auto-label review", label: "Edit suggestion", description: "Open the only pending suggestion for correction.", defaultKey: "E", contexts: idle },
  { id: "verifyNegative", group: "No-box review", label: "Verify negative", description: "Confirm the current image has no target objects in Select image → Review no-box.", defaultKey: "V", contexts: ["negative-review"] },
  ...Array.from({ length: 9 }, (_, index): ShortcutDefinition => ({
    id: `class${index + 1}`, group: "Object Detection", label: `Select class ${index + 1}`,
    description: `Select class in position ${index + 1} of the Classes list, if it exists.`,
    defaultKey: String(index + 1), contexts: ["detect"],
  })),
  { id: "addPose", group: "Pose Estimation", label: "Add pose", description: "Start placing a new 17-point person pose.", defaultKey: "A", contexts: ["pose"] },
  { id: "poseVisible", group: "Pose Estimation", label: "Visible point", description: "Mark the next new pose point as visible before clicking the image.", defaultKey: "1", contexts: ["pose-drawing"] },
  { id: "poseOccluded", group: "Pose Estimation", label: "Occluded point", description: "Mark the next new pose point as occluded before clicking the image.", defaultKey: "2", contexts: ["pose-drawing"] },
  { id: "poseOutside", group: "Pose Estimation", label: "Outside frame", description: "Skip the next new pose point as outside the frame.", defaultKey: "3", contexts: ["pose-drawing"] },
  { id: "previousPoint", group: "Pose Estimation", label: "Previous keypoint", description: "Select the preceding keypoint while editing a pose.", defaultKey: "Q", contexts: ["pose-edit"] },
  { id: "nextPoint", group: "Pose Estimation", label: "Next keypoint", description: "Select the following keypoint while editing a pose.", defaultKey: "E", contexts: ["pose-edit"] },
];

export type ShortcutBindings = Record<string, string>;

export const DEFAULT_SHORTCUTS: ShortcutBindings = Object.fromEntries(
  SHORTCUT_DEFINITIONS.map(({ id, defaultKey }) => [id, defaultKey]),
);

export function shortcutFromEvent(event: KeyboardEvent | ReactKeyboardEvent): string | null {
  const key = event.key;
  if (["Control", "Meta", "Alt", "Shift", "Tab", "Dead", "Unidentified"].includes(key) || /^F\d+$/.test(key)) return null;
  const normalized = key.length === 1 ? key.toUpperCase() : key;
  const modifiers = [event.ctrlKey || event.metaKey ? "Mod" : "", event.altKey ? "Alt" : "", event.shiftKey ? "Shift" : ""].filter(Boolean);
  return [...modifiers, normalized === " " ? "Space" : normalized].join("+");
}

export function matchesShortcut(event: KeyboardEvent, bindings: ShortcutBindings, id: string): boolean {
  return !!bindings[id] && shortcutFromEvent(event) === bindings[id];
}

export function shortcutConflict(bindings: ShortcutBindings, id: string, key: string): ShortcutDefinition | null {
  if (!key) return null;
  const definition = SHORTCUT_DEFINITIONS.find((item) => item.id === id);
  return SHORTCUT_DEFINITIONS.find((item) => item.id !== id && bindings[item.id] === key &&
    item.contexts.some((context) => definition?.contexts.includes(context))) ?? null;
}

export function loadShortcuts(userId: number | undefined): ShortcutBindings {
  if (userId == null) return { ...DEFAULT_SHORTCUTS };
  try {
    const saved = JSON.parse(localStorage.getItem(`cv-platform.shortcuts.${userId}`) ?? "{}") as ShortcutBindings;
    return Object.fromEntries(SHORTCUT_DEFINITIONS.map(({ id, defaultKey }) => [id, typeof saved[id] === "string" ? saved[id] : defaultKey]));
  } catch {
    return { ...DEFAULT_SHORTCUTS };
  }
}

export function saveShortcuts(userId: number, bindings: ShortcutBindings): void {
  try { localStorage.setItem(`cv-platform.shortcuts.${userId}`, JSON.stringify(bindings)); } catch { /* Browser storage may be disabled. */ }
}
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
