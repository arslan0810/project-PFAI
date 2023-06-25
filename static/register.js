document.addEventListener("DOMContentLoaded", function() {
    var signupForm = document.querySelector("#signup-form");
    signupForm.addEventListener("submit", signupValidation);
});

function signupValidation(event) {
    event.preventDefault(); // Prevent form submission

    var valid = true;
    // ... existing validation code ...

    if (valid) {
        var username = document.querySelector("#username").value;
        var email = document.querySelector("#email").value;
        var password = document.querySelector("#signup-password").value;
        $.ajax({
            url: '/register',
            method: 'POST',
            data: {
                username: username,
                email: email,
                password: password
            },
            success: function(response) {
                console.log("Registration successful!");
            },
            error: function(error) {
                ("Registration failed:", response.statusText);
            }
        });
    }
}