// 드래그 & 드롭 + 미리보기 + 업로드 진행 표시
(function () {
  const dz = document.getElementById('drop-zone');
  const input = document.getElementById('image-input');
  const empty = document.getElementById('dz-empty');
  const preview = document.getElementById('dz-preview');
  const previewImg = document.getElementById('preview-img');
  const previewName = document.getElementById('preview-name');
  const form = document.getElementById('upload-form');
  const submitBtn = document.getElementById('submit-btn');
  if (!dz || !input) return;

  function showPreview(file) {
    if (!file) return;
    if (file.type && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => { previewImg.src = e.target.result; };
      reader.readAsDataURL(file);
    } else {
      previewImg.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="180"><rect width="100%" height="100%" fill="#1e293b"/><text x="50%" y="50%" fill="#94a3b8" text-anchor="middle" font-family="sans-serif" font-size="14">DICOM 파일</text></svg>'
      );
    }
    previewName.textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    empty.classList.add('hidden');
    preview.classList.remove('hidden');
  }

  input.addEventListener('change', () => showPreview(input.files[0]));

  ['dragenter', 'dragover'].forEach(evt =>
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add('is-dragover'); })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove('is-dragover'); })
  );
  dz.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    input.files = e.dataTransfer.files;
    showPreview(file);
  });

  if (form && submitBtn) {
    form.addEventListener('submit', () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>업로드 후 AI 분석 중...';
    });
  }
})();
