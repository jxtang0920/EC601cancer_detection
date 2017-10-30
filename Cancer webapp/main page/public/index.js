
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

//welcome alert------------------------
   $("#wel").click(
   	function(){

   		alert("welcome");
	}

   	);
//button----------------
 $("#news").click(
   	function(){

   		//alert("welcome");
		window.open("https://www.google.com/search?q=cancer+detection+about+genes&source=lnms&sa=X&ved=0ahUKEwjivcrBkYXXAhUK2oMKHWLaA0kQ_AUICSgA&biw=1463&bih=718&dpr=1.75");
	});

 $("#contact").click(
   	function(){

   		//alert("welcome");
		window.open("mailto:caoliyi@bu.edu?subject=Hello%20again");
	});


//result------------------------------
 $("#result").click(
   	function(){

   		alert("result");
		
	});

 //updat dialog--------------------------------
   $("#Update").click(
   	function(){
   		window.open("https://cancer-update.firebaseapp.com");
  

   	});