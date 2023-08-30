let mode = 0;

function createOptions(nodes){
    const nodeDatasets = document.getElementsByClassName("nodeDataset");
    for(let i = 0; i < nodeDatasets.length; i++){
        const option = document.createElement("option");
        option.value = nodeDatasets[i].children[0].value;
        option.innerText = nodeDatasets[i].children[0].value;
        nodes.appendChild(option);
    }
}

function addFileInput(){
    const toAdd = document.createElement("div");
    toAdd.className = "associationFile";
    const fileInput = document.createElement("input");
    fileInput.type = 'file';
    fileInput.accept = '.txt';
    fileInput.name = "afiles";
    fileInput.required = true;
    const delImg = document.createElement("img");
    delImg.addEventListener("click", delFileInput);
    delImg.className="delFile";
    delImg.src="static/img/x.png";
    delImg.title="remove file from list";
    delImg.alt="del file";
    const label1 = document.createElement("label");
    label1.innerText = "Nodes Left:";
    const nodes1 = document.createElement("select");
    nodes1.className = 'nodes';
    nodes1.name = "nodes.left";
    createOptions(nodes1);

    const label2 = document.createElement("label");
    label2.innerText = "Nodes Right:";
    const nodes2 = document.createElement("select");
    nodes2.className = 'nodes';
    nodes2.name = "nodes.right";
    createOptions(nodes2);

    const label3 = document.createElement("label");
    label3.innerText = "Main:";
    const mainInput = document.createElement("input");
    mainInput.type = "radio";
    mainInput.name = "main";
    mainInput.value = document.getElementsByClassName("associationFile").length.toString();

    if(mode === 0){
      nodes2.hidden = true;
      nodes1.hidden = true;
      label2.hidden = true;
      label1.hidden = true;
      label3.hidden = true;
      mainInput.hidden = true;
    }

    toAdd.appendChild(fileInput);
    toAdd.appendChild(label1);
    toAdd.appendChild(nodes1);
    toAdd.appendChild(label2);
    toAdd.appendChild(nodes2);
    toAdd.appendChild(label3);
    toAdd.appendChild(mainInput);
    toAdd.appendChild(delImg);

    const addButton = document.getElementById("addFile")
    addButton.insertAdjacentElement('beforebegin',toAdd);

    // prevents form to submit
    return false
}

function delFileInput(){
    // get  child: divInput - Associations files
    const associationDiv = document.getElementById("associationFiles");
    const imgs = associationDiv.getElementsByTagName("img");
    const associationFiles = associationDiv.getElementsByClassName("associationFile");
    let value = 0;
    for(let i=0; i <= imgs.length; i++){
        if (imgs[i] === this) {
            // remove
            const toDel = associationFiles[i];
            associationDiv.removeChild(toDel);
            break;
        }else{
            associationFiles[i].children[6].value = value.toString()
            value++;
        }
    }

    // prevents form to submit
    return false;
}

function swapDivSFile(mod){
    mode = mod;
    const divL = document.getElementById("sfileLoaded");
    const divC = document.getElementById("sfileConfig");
    if(mod === 0){
        divL.hidden = false;
        divC.hidden = true;
    }
    else{
        divL.hidden = true;
        divC.hidden = false;
    }


    let associationFiles = document.getElementsByClassName("associationFile");
    for(let i = 0; i < associationFiles.length; i++){
        for(let j = 1; j < associationFiles[i].children.length - 1; j++){
            associationFiles[i].children[j].hidden = (mode === 0);
        }
    }
}

function showRankOptions(){
    const select = document.getElementById("initialization");
    let svdRankInput = document.getElementById("svdRank");
    let labelSvd = svdRankInput.previousElementSibling;
    let ks = document.getElementsByClassName("ks");
    let labelsK = document.getElementsByClassName("labelK");


    /*
        Uncomment for fast SVD
    if(select.value === "svd"){
        svdRankInput.hidden = false;
        labelSvd.hidden = false;
        for(let i = 0; i < ks.length; i++){
            ks[i].hidden = true;
            labelsK[i].hidden = true;
        }
    }
    else{
        svdRankInput.hidden = true;
        labelSvd.hidden = true;
        /*for(let i = 0; i < ks.length; i++){
            ks[i].hidden = false;
            labelsK[i].hidden = false;
        }
    }*/

    return false;
}

function addNodesDataset(){

    const toAdd = document.createElement("div");
    toAdd.className = "nodeDataset";
    const inputName = document.createElement("input");
    inputName.type = "text";
    inputName.required = true;
    inputName.name = "dsnames";
    inputName.placeholder = "nodes name";
    inputName.required = true;
    const label = document.createElement("label");
    label.innerText = "k:";
    label.title = "rank";
    label.className = "labelK";
    const inputK = document.createElement("input");
    inputK.type = "number";
    inputK.className = "ks";
    inputK.min = "1";
    inputK.max = "200";
    inputK.step = "1";
    inputK.name = "ks";
    inputK.value = "20";
    const select = document.getElementById("initialization");
    /*
    Uncomment for fast SVD
    if(select.value === "svd"){
        inputK.hidden = true;
        label.hidden = true;
    }*/
    const img = document.createElement("img");
    img.src = "static/img/x.png";
    img.className = "delFile";
    img.title = "remove nodes dataset";
    img.alt = "del nodes"
    img.addEventListener("click", delNodesDataset);

    toAdd.appendChild(inputName);
    toAdd.appendChild(label);
    toAdd.appendChild(inputK);
    toAdd.appendChild(img);
    toAdd.appendChild(document.createElement("br"));
    toAdd.appendChild(document.createElement("br"));

    const addButton = document.getElementById("addDataset")
    addButton.insertAdjacentElement('beforebegin',toAdd);

    return false;
}


function delNodesDataset(){
    const toDel = this.parentNode;

    document.getElementById("ranks").removeChild(toDel);
    updateFilesDatasets();

    // prevents form to submit
    return false;
}


function updateFilesDatasets(){
    const nodes = document.getElementsByClassName("nodes");
    for(let i = 0; i < nodes.length; i++) {
        nodes[i].innerHTML = "";
        createOptions(nodes[i]);
    }

    return false;
}