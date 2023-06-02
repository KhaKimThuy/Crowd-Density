$(document).ready(function(){
    const ip = document.getElementById("ip").textContent.concat(":9099/admin");
    const socket = io.connect(ip);
    const more_Btn = document.getElementById("more");

    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('monitor_request', function(sid) {
        console.log('monitor connected: '+sid.sid);

        // Create field for showcase
        const box_field = document.createElement("div");
        box_field.setAttribute("class", "box");
        box_field.setAttribute("id", sid.sid+"box");

        // Create field for monitor id
        const box_text = document.createElement('div')
        box_text.setAttribute("class", "box-text")
        const level = document.createElement('h3');
        level.setAttribute("id", sid.sid+"sidnumber");
        box_text.appendChild(level);
        box_field.appendChild(box_text);

        // Create field for only video
        const showcase = document.createElement('img');
        showcase.setAttribute("id", sid.sid);
        showcase.setAttribute("class", "monitor");
        box_field.appendChild(showcase);

        const killBtn = document.createElement("button");
        killBtn.setAttribute('class', 'btn btn-danger kill');
        killBtn.setAttribute('id', sid.sid+"kill");
        killBtn.onclick = () => {
            console.log("Button "+killBtn.id+" clicked!");
            socket.emit('cancel_task', {'client_id':killBtn.id});
            socket.emit('disconnect');
        }

        const client = document.createElement('p');
        client.setAttribute("id", sid.monitor);
        client.hidden = true;
        box_field.appendChild(client);

        killBtn.innerText = 'Terminate';
        box_field.appendChild(killBtn);

        document.getElementById('display').appendChild(box_field);
    });

    socket.on('estimate', function(msg) {

        // Set id monitor
        var monitor_code = document.getElementById(msg.sid+"sidnumber");
        monitor_code.innerText = "Monitor: " + msg.name_m;

        // Transmit estimated frame
        var image_element = document.getElementById(msg.sid);
        image_element.src="data:image/jpeg;base64," + msg.obs;

        // Transmit estimated frame
        var dens_element = document.getElementById(msg.name_m);
        dens_element.innerText = msg.dens;

    });

    socket.on('disconnect', function() {
        console.log('Client disconnected');
      });

    socket.on('task_cancelled', function(sid) {
        const element = document.getElementById(sid.sid_end+'box');
        element.remove();
    });

    more_Btn.onclick = () => {
        // window.location.replace("http://" + document.getElementById("ip").textContent + ":9099");
        window.open("http://" + document.getElementById("ip").textContent.concat(":9099"));
    }


    // Handle kill thread
    // const kill_Btn = document.getElementsByClassName("kill");
    // console.log('list thread id: '+kill_Btn.length)
    // for (var i = 0; i < kill_Btn.length; i++) {
    //     kill_Btn[i].addEventListener("click", function(event) {
    //         console.log('Handle kill thread '+event.target.id);
    //         socket.emit('close_tab', {'client_id':event.target.id})
    //     });
    // }
});