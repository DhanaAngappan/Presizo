var preview = document.getElementById('preview');

function displayFiles(files) {
    preview.innerHTML = '';

    for (var i = 0; i < files.length; i++) {
    var file = files[i];
    var reader = new FileReader();

    reader.onload = (function(file) {
        return function(e) {
        var fileType = file.type.split('/')[0];
        var previewItem = document.createElement('div');
        previewItem.className = 'preview-item';
        var previewElement;

        if (fileType === 'image') {
            previewElement = document.createElement('img');
        } else {
            return; // Unsupported file type
        }

        previewElement.src = e.target.result;
        previewItem.appendChild(previewElement);
        preview.appendChild(previewItem);
        };
    })(file);

    reader.readAsDataURL(file);
    }
}