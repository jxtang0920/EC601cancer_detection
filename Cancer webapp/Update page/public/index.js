
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

//input--------------------------



      var upcontent = document.getElementById('upcontent');
      var gene = document.getElementById("gene");
      var id = document.getElementById("id");
      var vari= document.getElementById("vari");
      var text = document.getElementById("text");

      //var upcontentRef = firebase.database().ref().child("Data"); 

      //upcontentRef.on('value'.function(datasnapshot){
         //upcontent.innertext = datasnapshot.val();
     // });\
   
   //var admin = require("firebase-admin");
   //admin.initializeApp(functions.config().firebase);

   var rootRef = firebase.database().ref().child('infos');
   $('#submitBtn').click(function(){
    rootRef.set({
      gene:$('gene').val(),
      id:$('id').val()
    });
   });