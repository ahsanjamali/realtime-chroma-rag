// Set the basic API address for communication with the backend server
const baseUrl = "http://localhost:8813";
// Flag indicating whether WebRTC is active, controls the enabling and disabling of connections
let isWebRTCActive = false;
// Create variables related to the WebRTC connection
let peerConnection;
let dataChannel;
// Add these variables at the top with other global variables
let isRecording = false;
// Define an object that contains multiple functions; methods in fns will be called
const fns = {
  // Add the search function
  search_knowledgebase: async ({ query }) => {
    try {
      const response = await fetch(`${baseUrl}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      return {
        success: true,
        results: data.results,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  },
};

// When an audio stream is received, add it to the page and play it
function handleTrack(event) {
  const el = document.createElement("audio"); // Create an audio element
  el.srcObject = event.streams[0]; // Set the audio stream as the element's source
  el.autoplay = el.controls = true; // Autoplay and display audio controls
  document.body.appendChild(el); // Add the audio element to the page
}

// Create a data channel for transmitting control messages (such as function calls)
function createDataChannel() {
  // Create a data channel named 'response'
  dataChannel = peerConnection.createDataChannel("response");

  // Configure data channel events
  dataChannel.addEventListener("open", () => {
    console.log("Data channel opened");
    configureData(); // Configure data channel functions
  });

  // Move the message event listener here inside createDataChannel
  dataChannel.addEventListener("message", async (ev) => {
    const msg = JSON.parse(ev.data);
    console.log("Received message type:", msg.type, "Full message:", msg);

    // Handle input audio transcription
    if (msg.type === "conversation.item.input_audio_transcription.completed") {
      console.log("Received audio transcript:", msg.transcript);
      addMessageToChat(msg.transcript, true); // true indicates it's a user message
    }

    // Handle function calls
    if (msg.type === "response.function_call_arguments.done") {
      const fn = fns[msg.name];
      if (fn !== undefined) {
        console.log(
          `Calling local function ${msg.name}, parameters ${msg.arguments}`
        );
        const args = JSON.parse(msg.arguments);
        const result = await fn(args);
        console.log("Result", result);
        const event = {
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: msg.call_id,
            output: JSON.stringify(result),
          },
        };
        dataChannel.send(JSON.stringify(event));
      }
      const createResponse = {
        type: "response.create",
        response: {
          modalities: ["text", "audio"],
        },
      };
      dataChannel.send(JSON.stringify(createResponse));
    }

    // Handle text responses
    if (msg.type === "response.text.delta") {
      console.log("Received text delta:", msg.delta);
      addMessageToChat(msg.delta, false);
    }

    // Handle final response output
    if (msg.type === "response.done") {
      console.log("Final response:", msg.response.output[0]);
      // Check if there's content in the output
      const output = msg.response.output[0];
      const outputContent = msg.response.output[0];
      console.log("Output content:", outputContent);
      if (output && output.content) {
        for (const content of output.content) {
          if (content.type === "text") {
            addMessageToChat(content.text, false);
          } else if (content.type === "audio" && content.transcript) {
            addMessageToChat(content.transcript, false);
          }
        }
      }
    }
  });
}

// Configure data channel functions and tools
function configureData() {
  console.log("Configuring data channel");
  const event = {
    type: "session.update",
    session: {
      instructions:
        "You are a knowledge base assistant.\n\nIn the tools you have the search tool to search through the knowledge base to find relevant information. Respond to the user in a friendly and helpful manner.",
      modalities: ["text", "audio"],
      turn_detection: null, // Disable VAD by default

      tools: [
        {
          type: "function",
          name: "search_knowledgebase",
          description:
            "Search through the knowledge base of documents uploaded by the user",
          parameters: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "The search query to find relevant information",
              },
            },
            required: ["query"],
          },
        },
      ],
    },
  };
  dataChannel.send(JSON.stringify(event));
}

// Get the control button element
const toggleButton = document.getElementById("toggleWebRTCButton");
// Add a click event listener to the button to toggle the WebRTC connection state
toggleButton.addEventListener("click", () => {
  // If WebRTC is active, stop the connection; otherwise, start WebRTC
  if (isWebRTCActive) {
    stopWebRTC(); // Stop WebRTC
    toggleButton.textContent = "start"; // Update button text
  } else {
    startWebRTC(); // Start WebRTC
    toggleButton.textContent = "stop"; // Update button text
  }
});

// Capture microphone input stream and initiate WebRTC connection
function startWebRTC() {
  // If WebRTC is already active, return directly
  if (isWebRTCActive) return;
  // Create a new peerConnection object to establish a WebRTC connection
  peerConnection = new RTCPeerConnection();
  peerConnection.ontrack = handleTrack; // Bind audio stream processing function
  createDataChannel(); // Create data channel
  // Request user's audio stream
  navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    // Add each track from the audio stream to the peerConnection
    stream
      .getTracks()
      .forEach((track) =>
        peerConnection.addTransceiver(track, { direction: "sendrecv" })
      );
    // Create an offer for the local connection
    peerConnection.createOffer().then((offer) => {
      peerConnection.setLocalDescription(offer); // Set local description (offer)
      // Send the offer to the backend for signaling exchange
      fetch(baseUrl + "/api/rtc-connect", {
        method: "POST",
        body: offer.sdp, // Send the SDP of the offer to the backend
        headers: {
          "Content-Type": "application/sdp",
        },
      })
        .then((r) => r.text())
        .then((answer) => {
          // Get the answer returned by the backend and set it as the remote description
          peerConnection.setRemoteDescription({ sdp: answer, type: "answer" });
        });
    });
  });
  // Mark WebRTC as active
  isWebRTCActive = true;
}

// Stop the WebRTC connection and clean up all resources
function stopWebRTC() {
  // If WebRTC is not active, return directly
  if (!isWebRTCActive) return;
  // Stop the received audio tracks
  const tracks = peerConnection
    .getReceivers()
    .map((receiver) => receiver.track);
  tracks.forEach((track) => track.stop());
  // Close the data channel and WebRTC connection
  if (dataChannel) dataChannel.close();
  if (peerConnection) peerConnection.close();
  // Reset connection and channel objects
  peerConnection = null;
  dataChannel = null;
  // Mark WebRTC as not active
  isWebRTCActive = false;
}

// Add these new functions and event listeners after the existing code
const textInput = document.getElementById("textInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");

// Function to add a message to the chat display
function addMessageToChat(text, isUser = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${
    isUser ? "user-message" : "assistant-message"
  }`;
  messageDiv.textContent = text;
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to send text message to the model
async function sendTextMessage(text) {
  if (!isWebRTCActive || !dataChannel) {
    console.log("WebRTC status:", isWebRTCActive, "dataChannel:", dataChannel);
    alert("Please start the connection first");
    return;
  }

  console.log("Sending message:", text);
  // Add user message to chat
  addMessageToChat(text, true);

  // Create conversation item with text
  const createMessage = {
    type: "conversation.item.create",
    item: {
      type: "message",
      role: "user",
      content: [
        {
          type: "input_text",
          text: text,
        },
      ],
    },
  };

  // Send the message
  dataChannel.send(JSON.stringify(createMessage));

  // Request response from the model
  const createResponse = {
    type: "response.create",
    response: {
      modalities: ["text", "audio"],
    },
  };

  dataChannel.send(JSON.stringify(createResponse));
}

// Handle send button click
sendButton.addEventListener("click", () => {
  const text = textInput.value.trim();
  if (text) {
    sendTextMessage(text);
    textInput.value = "";
  }
});

// Handle enter key in input
textInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    const text = textInput.value.trim();
    if (text) {
      sendTextMessage(text);
      textInput.value = "";
    }
  }
});

// Add mic button functionality
const micButton = document.getElementById("micButton");

micButton.addEventListener("click", () => {
  if (!isWebRTCActive || !dataChannel) {
    alert("Please start the connection first");
    return;
  }

  isRecording = !isRecording;
  micButton.classList.toggle("recording");

  if (isRecording) {
    // Enable VAD when mic button is clicked
    const enableVAD = {
      type: "session.update",
      session: {
        turn_detection: {
          type: "server_vad", // or other VAD settings as needed
        },
      },
    };
    dataChannel.send(JSON.stringify(enableVAD));
  } else {
    // Disable VAD when mic button is clicked again
    const disableVAD = {
      type: "session.update",
      session: {
        turn_detection: null,
      },
    };
    dataChannel.send(JSON.stringify(disableVAD));
  }
});
