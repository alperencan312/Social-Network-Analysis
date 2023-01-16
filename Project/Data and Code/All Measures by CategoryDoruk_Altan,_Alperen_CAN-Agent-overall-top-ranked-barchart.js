Chart.defaults.plugins.legend.position = 'bottom'

var data = {
labels:['84','82','99','86','74','76','75','14','77','73',],
datasets : [
{
label: 'Agent (Doruk Altan, Alperen CAN)',
backgroundColor: 'rgba(255,0,0,0.2)',
borderColor: 'rgba(255,0,0,1)',
borderWidth: 1,
data : [1,0.937055,0.921552,0.919402,0.888469,0.882684,0.877371,0.827648,0.812425,0.812361,]
},
]
}
var context = document.getElementById('All Measures by CategoryDoruk_Altan,_Alperen_CAN-Agent-overall-top-ranked-barchart').getContext("2d");
var chart = new Chart(context, {
		type: 'bar',
		data: data,
		options: {
			indexAxis:'y',
			autowidth:false,
			scales: {
			yAxes: [{
				ticks: {
					beginAtZero:true
				}
			}]
		}
	}
});

