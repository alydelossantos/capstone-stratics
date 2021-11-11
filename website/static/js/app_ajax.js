$(document).ready(function() {
  var table = $('#example').DataTable({
    "ajax": 'array.json',
    "deferRender": true
  });
});
