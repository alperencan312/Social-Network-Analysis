$(document).ready(function() {
	// replace <br> with vertical bar so that it exports well, and then strip other html characters
	var convertHtml = function ( data, row, column ) { return data.replace(/<br>/gi, "|").replace(/<(.|\n)*?>/g, ""); }

	$('table.measures-ranked').DataTable({
		dom: 'fBrtip',
		buttons: [
			// copy button
			{
				extend: 'copyHtml5',
				title: 'ranked-table-data',
				filename: 'ranked-table-data',
				exportOptions: {format: { body: convertHtml }}
			},
			// csv output button
			{
				extend:'csvHtml5',
				title: 'ranked-table-data',
				text: 'Print',
				filename: 'ranked-table-data',
				exportOptions: {format: { body: convertHtml }}
			}
		]	
	});
});
