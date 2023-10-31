/**
 * Multi-track mixer
 *
 * @see https://github.com/katspaugh/wavesurfer-multitrack
 */

var stems = ["vocals", "drums", "bass", "other"]
// var metaData = [
//   {"name": "Vocals", "progressColor": "hsl(206, 100%, 65%)"},
//   {"name": "Drums", "progressColor": "hsl(161, 87%, 65%)"},
//   {"name": "Bass", "progressColor": "hsl(354, 100%, 65%)"},
//   {"name": "Other", "progressColor": "hsl(60, 100%, 65%)"}
// ]
const ctx = document.createElement('canvas').getContext('2d')
const gradient = ctx.createLinearGradient(0, 0, 0, 150)
gradient.addColorStop(0, 'rgb(42, 153, 218)')
gradient.addColorStop(0.5, 'rgb(26, 95, 134)')
gradient.addColorStop(1, 'rgb(14, 49, 70)')


const gradient1 = ctx.createLinearGradient(0, 0, 0, 150)
gradient1.addColorStop(0, 'rgb(11, 233, 196)')
gradient1.addColorStop(0.5, 'rgb(5, 128, 107)')
gradient1.addColorStop(1, 'rgb(2, 53, 44)')


const gradient2 = ctx.createLinearGradient(0, 0, 0, 150)
gradient2.addColorStop(0, 'rgb(238, 168, 77)')
gradient2.addColorStop(0.5, 'rgb(221, 164, 40)')
gradient2.addColorStop(1, 'rgb(102, 71, 30)')

const gradient3 = ctx.createLinearGradient(0, 0, 0, 150)
gradient3.addColorStop(0, 'rgb(200, 0, 200)')
gradient3.addColorStop(0.5, 'rgb(110, 1, 110)')
gradient3.addColorStop(1, 'rgb(78, 4, 78)')

var metaData = [
  {"stem": "Vocals", "progressColor": gradient},
  {"stem": "Drums", "progressColor": gradient1},
  {"stem": "Bass", "progressColor": gradient2},
  {"stem": "Other", "progressColor": gradient3}
]


const videoHeight = 340;
// console.log(111, `/static/audio/${songData.filename}.wav`)

const initMultiTrack = (filename) => {
  trackData = []
  for (var i = 0; i < metaData.length; i++) {
    track = {
      id: i,
      draggable: true,
      startPosition: 0.5,
      startCue: 0.1,
      volume: 0.5,
      options: {
        height: videoHeight/4,
        waveColor: "#656666", // 'hsl(206, 100%, 65%)',
        progressColor: metaData[i]["progressColor"],
      },
      url: `/static/audio/${filename}/${metaData[i]["stem"].toLowerCase()}.wav`,
      intro: {
        endTime: 200,
        label: metaData[i]["stem"],
        color: '#FFE56E',
      }
    }
    trackData.push(track)
  }

  const multitrack = Multitrack.create(trackData,
    {
      container: document.querySelector('#multitrack'), // required!
      minPxPerSec: 12, // zoom level
      rightButtonDrag: true, // drag tracks with the right mouse button
      cursorColor: '#57BAB6',
      cursorWidth: 5,
      trackBackground: '#2D2D2D',
      trackBorderColor: '#7C7C7C',
      height: 50,
      barHeight:1,
      envelopeOptions: {
        lineColor: 'rgba(255, 0, 0, 0.7)',
        lineWidth: 0,
        dragPointSize: 0,
        dragPointFill: 'rgba(255, 255, 255, 0.8)',
        dragPointStroke: 'rgba(255, 255, 255, 0.3)',
      },
    },
  )
  multitrack.on('volume-change', ({ id, volume }) => {
    console.log(`Track ${id} volume updated to ${volume}`)
  })
  const numTrack = multitrack.wavesurfers
  // Play/pause button
  const button = document.querySelector('#play')

  // Forward/back buttons
  const forward = document.querySelector('#forward')

  const backward = document.querySelector('#backward')



  var videoPlayer = document.querySelector("#video-player");
  videoPlayer.muted = true;


  button.disabled = true
  multitrack.once('canplay', () => {
    forward.onclick = () => {
      multitrack.setTime(multitrack.getCurrentTime() + 30);
      syncTrackToVideo()
    }
    backward.onclick = () => {
      multitrack.setTime(multitrack.getCurrentTime() - 30);
      syncTrackToVideo()
    }
    button.disabled = false;
    button.onclick = () => {
      multitrack.isPlaying() ? multitrack.pause() : multitrack.play()
      // button.textContent = multitrack.isPlaying() ? 'Pause' : 'Play'
      if (multitrack.isPlaying()) {
        button.classList.add("playing");
        videoPlayer.play();
        // player.playVideo()
      } else {
        button.classList.remove("playing");
        videoPlayer.pause();
        // player.pauseVideo()
      }
    }
  })





  // Destroy the plugin on unmount
  // This should be called before calling initMultiTrack again to properly clean up
  window.onbeforeunload = () => {
    multitrack.destroy()
  }



  const multitrack_ctl = document.querySelector('.multitrack-ctl')
  for (let i = 0; i < 4; i++) {
    const ctlDiv = document.createElement('div');
    ctlDiv.className = "play-btn btn"

    const stemImg = document.createElement("img")
    stemImg.src = `/static/images/instruments/${stems[i]}.png`
    stemImg.className = "stem-icon mute-btn"
    stemImg.setAttribute("width", "22px")
    
    const span = document.createElement('span');
    span.className = "mute-btn btn"
    span.setAttribute("data-track-id", i)


    const volUpIcon = document.createElement('i');
    volUpIcon.className ="fas fa-volume-up"
    span.appendChild(volUpIcon)

    const muteIcon = document.createElement('i');
    muteIcon.className ="fas fa-volume-mute"
    span.appendChild(muteIcon)

    // volume slider
    const volumeSlider = document.createElement('input');
    volumeSlider.className ="volume-slider"
    volumeSlider.setAttribute("type", "range")
    volumeSlider.setAttribute("min", 0.0)
    volumeSlider.setAttribute("max", 1.0)
    volumeSlider.setAttribute("step", 0.1)
    volumeSlider.setAttribute("value", 0.5)
    volumeSlider.setAttribute("data-track-id", i)
    ///////////

    ctlDiv.appendChild(stemImg)
    //ctlDiv.appendChild(span)
    ctlDiv.appendChild(volumeSlider)
    multitrack_ctl.appendChild(ctlDiv)


    
  }


  const playBtn = document.querySelector(".play-btn");
  const stopBtn = document.querySelector(".stop-btn");
  const muteBtns= document.querySelectorAll(".mute-btn");
  const volumeSliders = document.querySelectorAll(".volume-slider");

  for (var i = 0; i < volumeSliders.length; i++) {
    const volumeSlider = volumeSliders[i]
    const muteBtn = muteBtns[i]
    const trackId = volumeSlider.dataset.trackId
    volumeSlider.addEventListener("mouseup", () => {
      changeVolume(trackId, volumeSlider.value, muteBtn);
    });
    muteBtn.addEventListener("click", () => {
      var volume;
      if (muteBtn.classList.contains("muted")) {
        muteBtn.classList.remove("muted");
        volumeSlider.value = 0.5;
        volume = 0.5;
        
      } else {
        muteBtn.classList.add("muted");
        volumeSlider.value = 0;
        volume = 0;
      }

      if (trackId < 0){
        masterIsMute = muteBtns[0].classList.contains("muted")
        
        for (var i = 1; i < muteBtns.length; i++) {
          const muteBtn = muteBtns[i];
          const volumeSlider = volumeSliders[i]
          
          
          if (masterIsMute){
            muteBtn.classList.add("muted");
            volumeSlider.value = 0;
            multitrack.wavesurfers.forEach((wavesurfer) => {
              container = wavesurfer["renderer"]["container"]
              container.classList.add("disabletrack");
            });
            
            
          }else{
            muteBtn.classList.remove("muted");
            volumeSlider.value = 0.5;
            multitrack.wavesurfers.forEach((wavesurfer) => {
              container = wavesurfer["renderer"]["container"]
              container.classList.remove("disabletrack");
            });
          }
          
        }
        multitrack.wavesurfers.forEach((wavesurfer) => wavesurfer.setVolume(volume));
      } else{
        container = multitrack.wavesurfers[trackId]["renderer"]["container"]
        container.classList.add("disabletrack");
        multitrack.wavesurfers[trackId].setVolume(volume);
        isMutes = Array.prototype.map.call(Array.prototype.slice.call(muteBtns, 1), (btn) => btn.classList.contains("muted"))
        allIsMute = isMutes.reduce((partialBool, e) => partialBool & e);

        if (allIsMute){
            muteBtns[0].click();
        }
        else{
          if (!muteBtn.classList.contains("muted")) {
            muteBtns[0].classList.remove("muted");
            volumeSliders[0].value = 0.5;

            container = multitrack.wavesurfers[trackId]["renderer"]["container"];
            container.classList.remove("disabletrack");
            
          } 
        }
        
      }
    });
  }


  const changeVolume = (trackId, volume, muteBtn) => {
    console.log(trackId, volume, muteBtn)

    if (trackId < 0){
      multitrack.wavesurfers.forEach((wavesurfer) => {
        wavesurfer.setVolume(volume);
        if (volume <= 0) {
          container = wavesurfer["renderer"]["container"]
          container.classList.add("disabletrack");
        } else {
          
          container = wavesurfer["renderer"]["container"]
          console.log(container)
          container.classList.remove("disabletrack");
        }
      
      });
      for (var i = 0; i < volumeSliders.length; i++) {
        const volumeSlider = volumeSliders[i];
        const muteBtn = muteBtns[i];
        volumeSlider.value = volume;
        if (volume <= 0) {
          muteBtn.classList.add("muted");
        } else {
          muteBtn.classList.remove("muted");
        }

      }
    } else{
      multitrack.wavesurfers[trackId].setVolume(volume);
      if (volume <= 0) {
        muteBtn.classList.add("muted");
        container = multitrack.wavesurfers[trackId]["renderer"]["container"]
        container.classList.add("disabletrack");
      } else {
        muteBtn.classList.remove("muted");
        container = multitrack.wavesurfers[trackId]["renderer"]["container"]
        container.classList.remove("disabletrack");
      }
    }
    
  };
  const syncTrackToVideo = () => {
    currentTime = multitrack.getCurrentTime();
    videoPlayer.currentTime = currentTime;
    multitrack.setTime(currentTime);
    console.log(currentTime);
  };
  trackContainer = document.querySelector('#multitrack');
  trackContainer.onclick = () => {
    syncTrackToVideo()
    // currentTime = multitrack.getCurrentTime();
    // videoPlayer.currentTime = currentTime;
    // multitrack.setTime(currentTime);
    // console.log(currentTime);
    // videoPlayer.seekTo(currentTime);
    // player.playVideo();
    // console.log("video: ", player.getCurrentTime(), " - audio: ", multitrack.getCurrentTime())
    // 
  }

}


