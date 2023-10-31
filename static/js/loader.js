// Call Post API Handler
loader = document.querySelector(".loader-container");
console.log(loader)
submitBtn = document.querySelector("#submitBtn");
urlInput = document.querySelector("#url-input");
// submitBtn.addEventListener("click", fetchHandler);

function displayLoading() {
    loader.classList.remove("hidden");
    // to stop loading after some time
    // setTimeout(() => {
    //     loader.classList.add("hidden");
    // }, 180000);
}

// hiding loading 
function hideLoading() {
  loader.classList.add("hidden");
}

const fetchHandler = () => {
    resultDiv = document.querySelector(".result-container")
    messageDiv = document.querySelector(".message")
    resultDiv.innerHTML = ""
    messageDiv.innerHTML = ""
    displayLoading()
    
    var http = new XMLHttpRequest();
    var url = '/separate';
    var params = `url=${urlInput.value}`;
    http.open('POST', url, true);

    //Send the proper header information along with the request
    http.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');

    http.onreadystatechange = function() {//Call a function when the state changes.
        if(http.readyState == 4) {
            if (http.status == 200){
                // alert(http.responseText);
                hideLoading()
                result = JSON.parse(http.responseText);
                resultDiv.innerHTML = result["innerHTML"];
                initMultiTrack(result["filename"]);
            }
            else if (http.status == 401){
                hideLoading()
                messageContent = `<h1 style="text-align: center; margin-top: 10px; color:red">${http.responseText}</h1>`
                messageDiv.innerHTML = messageContent;
            }
            
        }
    }
    http.send(params);
        
};

submitBtn.onclick = () => {
  fetchHandler()
}

urlInput.addEventListener("keypress", function(event) {
    // If the user presses the "Enter" key on the keyboard
    if (event.key === "Enter") {
        fetchHandler()
    }
});