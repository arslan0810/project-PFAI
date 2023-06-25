const textarea = document.getElementById('message-input');
const button_send = document.querySelector('#send-button');
const chatbox = document.querySelector(".chat-container");

button_send.addEventListener('click', (e) => {
    e.preventDefault();
    let text_Value = textarea.value;
    textarea.value = '';
    console.log(text_Value)

    const newElement = document.createElement('div');
    newElement.classList.add('chat-message');
    newElement.classList.add('user-message');
    newElement.innerHTML = `<div class="message">${text_Value}</div>`;
    chatbox.appendChild(newElement);


    $.ajax({
        url: '/process',
        method: 'POST',
        data: {
            user_input: text_Value,
        },
        success: function(response) {
            const botnewElement = document.createElement('div');
            botnewElement.classList.add('chat-message');
            botnewElement.classList.add('bot-message');
            botnewElement.innerHTML = `<div class="message">${response}</div>`;
            chatbox.appendChild(botnewElement);
            $(chatbox).animate({
                scrollTop: chatbot.scrollHeight * 10
            });
        },
        error: function(error) {
            const botnewElement = document.createElement('div');
            botnewElement.classList.add('chat-message');
            botnewElement.classList.add('bot-message');
            botnewElement.innerHTML = `<div class="message">${error}</div>`;
            chatbox.appendChild(botnewElement);
        }
    });


});