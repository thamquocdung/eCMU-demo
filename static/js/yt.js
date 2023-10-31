// 2. This code loads the IFrame Player API code asynchronously.
var tag = document.createElement('script');

tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

// 3. This function creates an <iframe> (and YouTube player)
//    after the API code downloads.
var player;
function onYouTubeIframeAPIReady() {
  player = new YT.Player('player', {
    height: '390',
    width: '640',
    videoId: 'W4P8gl4dnrg',
    playerVars: {
      'playsinline': 1
    },
    events: {
      'onReady': onPlayerReady,
      'onStateChange': onPlayerStateChange
    }
  });
}

// 4. The API will call this function when the video player is ready.
function onPlayerReady(event) {
    event.target.setVolume(0);
    // event.target.playVideo();
}

// 5. The API calls this function when the player's state changes.
//    The function indicates that when playing a video (state=1),
//    the player should play for six seconds and then stop.
var done = false;
// var currentTime = 0.05;
const trackBtn = document.querySelector("#play");
function onPlayerStateChange(event) {
  console.log("before: video: ", player.getCurrentTime(),  " - audio: ", multitrack.getCurrentTime())
  multitrack.setTime(player.getCurrentTime());
  console.log("after: video: ", player.getCurrentTime(),  " - audio: ", multitrack.getCurrentTime())
  if (event.data == YT.PlayerState.PLAYING) {
    
    // multitrack.setTime(player.getCurrentTime());
    
    if (!trackBtn.classList.contains("playing")){
        // currentTime = player.getCurrentTime();
        trackBtn.click();
        
    }
  }
  else if (event.data == YT.PlayerState.PAUSED || event.data == YT.PlayerState.BUFFERING){
    if (trackBtn.classList.contains("playing")){
        trackBtn.click();
        // currentTime = player.getCurrentTime();
        
    }
  }
}
function stopVideo() {
  player.stopVideo();
}