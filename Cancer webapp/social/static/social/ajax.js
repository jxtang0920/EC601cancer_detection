$(function(){

	$('#user').blur(function(){
		var csrf = "input[name=csrfmiddlewaretoken]";
		$.ajax({
			type: 'POST',
			url: '/social/checkuser/',
			data : {
				'user' : $('#user').val(),
				'csrfmiddlewaretoken' : $(csrf).val()
			},
			success: checkuseranswer,
			dataType: 'html'
		});
	});

});

function checkuseranswer(data, textStatus, jqHXR)
{
	$('#info').html(data);
}