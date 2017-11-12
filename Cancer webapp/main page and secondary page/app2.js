
function loadFile(input) {

        //Step 1 : Defining element to show the progress
        var elem = document.getElementById("myBar");    
        var filetoUpload=input.files[0];

        //Step 2 : Initializing the reference of database with the filename
        var storageRef = firebase.storage().ref(filetoUpload.name);
        //Step 3 : Uploading file
         var task = storageRef.put(filetoUpload);
        
        //Step 4 : sata_changed Event
         // state_changed events occures when file is getting uploaded 
         //(Note : when we want to show the progress what's the uploading status that time we will use this function.)
         task.on('state_changed',
            function progress(snapshot){
                var percentage = snapshot.bytesTransferred / snapshot.totalBytes * 100;
                //uploader.value = percentage;
                 elem.style.width = parseInt(percentage) + '%'; 
                 elem.innerHTML=parseInt(percentage)+'%';
            },
            function error(err){

            },
            function complete(){
                var downloadURL = task.snapshot.downloadURL;
            }
        ); 
}


//////////////////////////
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





/////////////////////////////////

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
       $("#loginDialog").hide();

  }
  );






     $("#logOutBtn").click(
     function(){
     firebase.auth().signOut().then(function() {
      // Sign-out successful.
    }).catch(function(error) {
      // An error happened.
    alert("error message");
    });
  
   location.reload()
   var x = document.getElementById("loginDialog");
   x.open = true;
   
   //location.reload()
    }
    );
