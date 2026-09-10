const dashboard = document.querySelector("#dashboard");
const workspace = document.querySelector("#workspace");
const reviewQueue = document.querySelector("#reviewQueue");
const dashboardButton = document.querySelector("#dashboardButton");
const saveButton = document.querySelector("#saveButton");
const toast = document.querySelector("#toast");

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 1600);
}

function showDashboard() {
  dashboard.classList.remove("hidden");
  workspace.classList.add("hidden");
  reviewQueue.classList.add("hidden");
}

function showWorkspace() {
  dashboard.classList.add("hidden");
  workspace.classList.remove("hidden");
  reviewQueue.classList.remove("hidden");
}

dashboardButton.addEventListener("click", showDashboard);
document.querySelectorAll("[data-open-workspace]").forEach((button) => button.addEventListener("click", showWorkspace));
saveButton.addEventListener("click", () => showToast("Mock label saved locally for preview"));

document.querySelectorAll("button:not(.brand):not([data-open-workspace]):not(#saveButton)").forEach((button) => {
  button.addEventListener("click", () => showToast("Mockup only - no API request was sent"));
});
