console.log("in dropdown")
$('.dropdown-toggle').dropdown();
$('#sortOptions').on('click', function() {
  $('#sortByTitle').html($(this).find('option').html());
});​