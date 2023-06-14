let clientSocket;

function openWebSocketCommunication(id){
    clientSocket = new WebSocket("ws://" + location.host + "/sock");
    console.log("Connection opened");

    clientSocket.onopen = (event) => {
        clientSocket.send(id);
        console.log("Websocket connection opened!");
        const prompt = document.getElementById("console");
        prompt.innerText += "\nWebsocket connection opened!";
    };

    clientSocket.onmessage = (event) => {
        if(event.data !== "close"){
            const prompt = document.getElementById("console");
            prompt.innerText += ("\n" + event.data);
            prompt.scrollTop = prompt.scrollHeight;
        }
        else{
            clientSocket.close(1000);
        }
    };

    clientSocket.onclose = function(event) {
        if (event.wasClean) {
            if(1000 === event.code){
                alert("Your results are ready!\ncode="+event.code+' reason='+event.reason)
                const prompt = document.getElementById("console");
                prompt.innerHTML += ("<br><a style='color: #07cb07' href='\showdata?name="+id+"'>Click here for your results</a>");
                prompt.scrollTop = prompt.scrollHeight;
            }
            else{
                alert("Something went wrong with your analysis!\ncode="+event.code+' reason='+event.reason)
            }
        }
        else{
            alert('[close] Connection died!\ncode='+event.code+' reason='+event.reason);
        }
        document.getElementById("divText").style.display = "none";
    };
}