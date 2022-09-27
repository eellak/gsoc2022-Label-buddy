// disable button if there are no utilities associated with this model
$(document).ready(
    function()
    {
        const send_approved_annotations_button = document.getElementById('send-approved-annotations-button');
        if (project_prediction_model_title == 'YOHO_container') send_approved_annotations_button.disabled = false;
        
        const training_button = document.getElementById('training-button');
        if (project_prediction_model_title == 'YOHO_container') training_button.disabled = false;

        const send_base_training_dataset_button = document.getElementById('send-base-training-dataset-button');
        if (project_prediction_model_title == 'YOHO_container') send_base_training_dataset_button.disabled = false;

        const send_base_validation_dataset_button = document.getElementById('send-base-validation-dataset-button');
        if (project_prediction_model_title == 'YOHO_container') send_base_validation_dataset_button.disabled = false;

        const dowload_model_weights_button = document.getElementById('dowload-model-weights-button');
        if (project_prediction_model_title == 'YOHO_container') dowload_model_weights_button.disabled = false;
    }
);

function send_approved_annotations(){

    let url = "http://127.0.0.1:5000/get_approved_data_annotations?project_id=" + project_id
    alert(url)
    let xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
           // Typical action to be performed when the document is ready:
            NProgress.done();
            alert("Approved Annotations sent!");
        } else if(this.readyState == 4 && (this.status == 400 || this.status == 401)){
            NProgress.done();
            alert("Approved Annotations failed to be sent!");
        }
    };

    xhr.open("POST", url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    NProgress.start();
    xhr.send(null);
}

$(document).ready(
    function()
    {
        document.getElementById('send-approved-annotations-button').addEventListener("click", function() {     
            send_approved_annotations()
        });
    }
);


function training(epochs, patience, initial_lr){

    let url = "http://127.0.0.1:5000/train?epochs=" + epochs + '&patience=' + patience + '&lr=' + initial_lr
    alert(url)
    let xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
           // Typical action to be performed when the document is ready:
            NProgress.done();
            alert("Training has begun!");
            data = JSON.parse(this.responseText);

            document.getElementById('binary_accuracy').innerHTML = data['binary_accuracy'];
            document.getElementById('loss').innerHTML = data['loss'];
            alert('Training done!')

        } else if(this.readyState == 4 && (this.status == 400 || this.status == 401)){
            NProgress.done();
            alert("Failed to start training process.");
        }
    };

    xhr.open("POST", url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    NProgress.start();
    xhr.send(null);
}

$(document).ready(
    function()
    {
        document.getElementById('train-modal-btn').addEventListener("click", function() {   
            let epochs = document.getElementById('epochNumber').value;
            let patience = document.getElementById('patienceNumber').value;
            let initial_lr = document.getElementById('lrNumber').value;

            if (epochs < 1){
                epochs = 300
            }
            
            if (patience < 1){
                patience = 15
            } 
            
            if (initial_lr < 0.00001 || initial_lr > 1){
                initial_lr = 0.001
            }  
            
            training(epochs, patience, initial_lr)
        });
    }
);

function send_base_training_dataset(){

    let url = "http://127.0.0.1:5000/get_training_data"
    alert(url)
    let xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
           // Typical action to be performed when the document is ready:
            NProgress.done();
            alert("Base training dataset has been sent!");
        } else if(this.readyState == 4 && (this.status == 400 || this.status == 401)){
            NProgress.done();
            alert("Failed to send base training Dataset.");
        }
    };

    xhr.open("POST", url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    NProgress.start();
    xhr.send(null);
}

$(document).ready(
    function()
    {
        document.getElementById('send-base-training-dataset-button').addEventListener("click", function() {     
            send_base_training_dataset()
        });
    }
);


function send_base_validation_dataset(){

    let url = "http://127.0.0.1:5000/get_validation_data"
    alert(url)
    let xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
           // Typical action to be performed when the document is ready:
            NProgress.done();
            alert("Base validtion dataset has been sent!");
        } else if(this.readyState == 4 && (this.status == 400 || this.status == 401)){
            NProgress.done();
            alert("Failed to send base validtion Dataset.");
        }
    };

    xhr.open("POST", url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    NProgress.start();
    xhr.send(null);
}

$(document).ready(
    function()
    {
        document.getElementById('send-base-validation-dataset-button').addEventListener("click", function() {     
            send_base_validation_dataset()
        });
    }
);