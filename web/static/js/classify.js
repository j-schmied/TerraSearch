// Function to add input text  to div with id metadata
function addMetadata() {
    // Erstellen eines input-Elements vom Typ "text"
    var input = document.createElement("input");
    input.type = "text";

    // Zugriff auf das "metadata" -Div-Element
    var metadataDiv = document.getElementById("metadata");

    // Hinzufügen des Input-Elements zum "metadata" -Div-Element
    metadataDiv.appendChild(input);
}

function addLocation() {
    // Erstellen eines input-Elements vom Typ "text"
    var input = document.createElement("input");
    input.type = "text";

    // Zugriff auf das "metadata" -Div-Element
    var metadataDiv = document.getElementById("location-info");

    // Hinzufügen des Input-Elements zum "metadata" -Div-Element
    metadataDiv.appendChild(input);
}