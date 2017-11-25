var config = {
    apiKey: "AIzaSyACzBBUAFoVMZ0CAuN0vBvKQ35Qd-51FZw",
    authDomain: "cancerdetection-b4ed9.firebaseapp.com",
    databaseURL: "https://cancerdetection-b4ed9.firebaseio.com",
    projectId: "cancerdetection-b4ed9",
    storageBucket: "",
    messagingSenderId: "304763830743"
  };
  firebase.initializeApp(config);

// Initialize Cloud Firestore through Firebase
var firestore = firebase.firestore();

const docRef = firestore.doc("samples/patientData");
const outputHeader = document.querySelector("#Output");
const inputTextField = document.querySelector("#MostRecentUpload");
const saveButton = document.querySelector("#saveButton");

saveButton.addEventListener("click", function(){
 const textToSave = inputTextField.value;
 console.log("Saving"+ textToSave + " to Firestore Database");
 docRef.set({
 UploadData: textToSave
 }).then(function(){
 console.log("Status saved!");
 }).catch(function(error) {
 console.log("Error:",error);
 });
});
