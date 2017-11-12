
//signin-------------------------
$("#signInBtn").click(
		function(){
			$(".login-cover").show();
		});

firebase.auth().onAuthStateChanged(function(user) {
	if(user){
		//user is signed in.

		//$(".login-cover").hide();
		//no user is signed in.
		var dialog = document.querySelector('#loginDialog');

		 if (! dialog.showModal) {
	      dialogPolyfill.registerDialog(dialog);
	    }
		dialog.close();

	}else{
		//no user is signed in.
		$(".login-cover").show();
		$("#loginProgress").hide();

		var dialog = document.querySelector('#loginDialog');

		 if (! dialog.showModal) {
		      dialogPolyfill.registerDialog(dialog);
		    }
			dialog.showModal();

	}
});

//*login process-----------------------------------
   $("#loginBtn").click(
	function(){
		var email= $("#loginEmail").val();
		var password = $("#loginPassword").val();

		if(email != "" && password != ""){


		$("#logBtn").hide();
		$("#message").hide();

		firebase.auth().signInWithEmailAndPassword(email,password).catch(function(error){
			$("#loginError").show().text(error.message);

			$("#loginProgress").hide();

			$("#logBtn").show();

		});
		
		}

	}
	);



   






/*sign up process----------------------------------------*/
   $("#signUpBtn").click(
	function(){
		var email= $("#loginEmail").val();
		var password = $("#loginPassword").val();

		if(email != "" && password != ""){


		$("#signUpBtn").hide();
		$("#message").hide();

		firebase.auth().createUserWithEmailAndPassword(email,password).catch(function(error){
			$("#loginError").show().text(error.message);

		

			$("#signUpBtn").show();

		});
		
		}

	}
	);




   /*logout process-------------------------*/
   $("#logOutBtn").click(
   	function(){

   		firebase.auth().signOut().then(function() {
		  // Sign-out successful.
		}).catch(function(error) {
		  // An error happened.
		alert("error message");
		});
	}

   	);



 //checkbox

  	function showPassword()
  	{
     
     var pass = document.getElementById('loginPassword');
   
     if(document.getElementById('checkbox').checked)
     {
     	pass.setAttribute('type','text');
     }
     	else{
     		pass.setAttribute('type','password');
     	}


	}



$("#adBtn").click(
	function(){
		var email= $("#loginEmail").val();
		var password = $("#loginPassword").val();

		if(email != "" && password != ""){


		$("#logBtn").hide();
		$("#message").hide();

		firebase.auth().signInWithEmailAndPassword(email,password).catch(function(error){
			$("#loginError").show().text(error.message);

			$("#loginProgress").hide();

			$("#logBtn").show();

		});
		window.location.href='update.html';
		
		}


	}
	);


