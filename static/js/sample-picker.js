// Sample picker: fetch a bundled static sample and place it into a file input.
(function () {
  async function chooseSample(button) {
    const target = document.querySelector(button.dataset.sampleTarget || "");
    const url = button.dataset.sampleUrl;
    const name = button.dataset.sampleName || (url ? url.split("/").pop() : "sample");
    if (!target || !url) return;

    button.disabled = true;
    const originalHtml = button.innerHTML;
    button.textContent = "불러오는 중...";

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`sample fetch failed: ${response.status}`);
      const blob = await response.blob();
      const file = new File([blob], name, { type: blob.type || "application/octet-stream" });
      const transfer = new DataTransfer();
      transfer.items.add(file);
      target.files = transfer.files;
      target.dispatchEvent(new Event("change", { bubbles: true }));

      const group = button.closest("[data-sample-picker]");
      group?.querySelectorAll("[data-sample-url]").forEach((btn) => {
        btn.classList.remove("border-brand-400", "bg-brand-50", "text-brand-700");
      });
      button.classList.add("border-brand-400", "bg-brand-50", "text-brand-700");
    } catch (error) {
      console.error(error);
      alert("샘플 파일을 불러오지 못했습니다. 직접 파일을 선택해주세요.");
    } finally {
      button.disabled = false;
      button.innerHTML = originalHtml;
    }
  }

  document.querySelectorAll("[data-sample-url]").forEach((button) => {
    button.addEventListener("click", () => chooseSample(button));
  });
})();
