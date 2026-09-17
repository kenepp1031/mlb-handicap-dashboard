"""Client-side "save as image" button for the day view.

Uses html2canvas (loaded from CDN on first click) to snapshot the full page
and trigger a normal browser download of a PNG. Runs
entirely in the browser -- no server round-trip and no extra dependencies.
"""

import streamlit as st

_HTML = """<button id="save-btn" type="button">Save day as image</button>"""

_CSS = """
#save-btn {
  font-family: var(--st-font, inherit);
  font-size: var(--st-font-size-sm, 0.875rem);
  color: var(--st-text-color, inherit);
  background-color: var(--st-secondary-background-color, transparent);
  border: 1px solid var(--st-border-color, rgba(128, 128, 128, 0.4));
  border-radius: var(--st-radius-default, 0.5rem);
  padding: 0.375rem 0.75rem;
  cursor: pointer;
  white-space: nowrap;
}
#save-btn:hover {
  border-color: var(--st-primary-color, currentColor);
  color: var(--st-primary-color, inherit);
}
#save-btn:disabled {
  opacity: 0.6;
  cursor: default;
}
"""

_JS = """
function loadHtml2Canvas() {
  if (window.html2canvas) return Promise.resolve()
  if (window.__html2canvasLoading) return window.__html2canvasLoading
  window.__html2canvasLoading = new Promise((resolve, reject) => {
    const script = document.createElement("script")
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"
    script.onload = () => resolve()
    script.onerror = () => reject(new Error("Failed to load html2canvas"))
    document.head.appendChild(script)
  })
  return window.__html2canvasLoading
}

export default function (component) {
  const { data, parentElement } = component
  const btn = parentElement.querySelector("#save-btn")
  if (!btn) return

  const defaultLabel = "Save day as image"
  btn.textContent = defaultLabel

  btn.onclick = async () => {
    btn.disabled = true
    btn.textContent = "Saving…"

    try {
      await loadHtml2Canvas()

      // The page now scrolls as one document (no inner clipped container),
      // so a plain full-document capture picks up everything, not just
      // what's currently in the viewport.
      const fullWidth = document.documentElement.scrollWidth
      const fullHeight = document.documentElement.scrollHeight
      const bg = window.getComputedStyle(document.body).backgroundColor

      const canvas = await window.html2canvas(document.body, {
        backgroundColor: bg,
        scale: 2,
        useCORS: true,
        windowWidth: fullWidth,
        windowHeight: fullHeight,
        scrollX: -window.scrollX,
        scrollY: -window.scrollY,
      })
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"))
      if (!blob) throw new Error("toBlob returned null")
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = data.filename || "mlb-day.png"
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
      btn.textContent = "Saved!"
    } catch (err) {
      console.error(err)
      btn.textContent = "Save failed"
    } finally {
      setTimeout(() => {
        btn.textContent = defaultLabel
        btn.disabled = false
      }, 2000)
    }
  }
}
"""

_SAVE_DAY_IMAGE = st.components.v2.component(
    "save_day_image",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def save_day_image_button(filename: str, *, key: str | None = None):
    """Render a button that screenshots the whole page and downloads it as `filename`."""
    return _SAVE_DAY_IMAGE(data={"filename": filename}, key=key)
