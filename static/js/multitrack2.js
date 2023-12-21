/**
 * Multi-track mixer
 *
 * @see https://github.com/katspaugh/wavesurfer-multitrack
 */

 var stems = ["vocals", "drums", "bass", "other"]

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
 
 const _addController = (multitrack_ctl) =>{
   for (let i = 0; i < 4; i++) {
     const ctlDiv = document.createElement('div');
     ctlDiv.className = "play-btn btn"
 
     const stemImg = document.createElement("img")
     stemImg.src = `./static/images/instruments/${stems[i]}.png`
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
 }
 const addController = () =>{
   const multitrack_ctl_lst = document.querySelectorAll('.multitrack-ctl')
   multitrack_ctl_lst.forEach(_addController)
   
 }
 
 const addEventMultitrack = (multitrack, index) => {
   
   multitrack.on('volume-change', ({ id, volume }) => {
     console.log(`Track ${id} volume updated to ${volume}`)
   })
   //var numTrack = multitrack.wavesurfers
   // Play/pause button
   var play = document.querySelectorAll('.play')[index] // button
 
   // Forward/back buttons
   var forward = document.querySelectorAll('.forward')[index]
   var backward = document.querySelectorAll('.backward')[index]
   
 
 
   var videoPlayer = document.querySelectorAll(".video-player")[index];
   videoPlayer.onseeked = (event) => {
     syncVideoToTrack();
   };
   // videoPlayer.muted = true;
   
 
   // play.disabled = true
   multitrack.once('canplay', () => {
     forward.onclick = () => {
       multitrack.setTime(multitrack.getCurrentTime() + 5);
       syncTrackToVideo()
     }
     backward.onclick = () => {
       multitrack.setTime(multitrack.getCurrentTime() - 5);
       syncTrackToVideo()
     }
     // play.disabled = false;
     play.onclick = () => {
       multitrack.isPlaying() ? multitrack.pause() : multitrack.play()
 
       if (multitrack.isPlaying()) {
         play.classList.add("playing");
         videoPlayer.play();
         // player.playVideo()
       } else {
         play.classList.remove("playing");
         videoPlayer.pause();
         // player.pauseVideo()
       }
     }
     // multitrack.wavesurfers[1].on('finish', () => {
     //   // multitrack.pause()
     //   // multitrack.setTime(0)
     //   console.log("finished")
     // })
     multitrack.wavesurfers[0].on('audioprocess', function (time) {
       pointer = $(`#multitrack${index} > div > div > div`)[0]
       waveProgress = parseFloat(pointer.style.left.slice(0,-1))
       if (!videoPlayer.muted){
         videoPlayer.muted = true
       }
       // console.log(time, multitrack.wavesurfers[0].getDuration(), parseFloat(pointer.style.left.slice(0,-1)))
       offset = Math.abs(100 - waveProgress)
       if (offset <= 0.05) {
         // console.log("finished");
         multitrack.pause()
         videoPlayer.pause()
         multitrack.setTime(0)
         videoPlayer.currentTime = 0
         multitrack.isPlaying() ? play.classList.add("playing") : play.classList.remove("playing")
       }
     });
     // $("#multitrack0 > div > div > div")[0].css()
   })
   
 
 
   // Destroy the plugin on unmount
   // This should be called before calling initMultiTrack again to properly clean up
   window.onbeforeunload = () => {
     multitrack.destroy()
   }
 
   const muteBtns= Array.from(document.querySelectorAll(".mute-btn")).slice(index*5, (index+1)*5);
   const volumeSliders = Array.from(document.querySelectorAll(".volume-slider")).slice(index*5, (index+1)*5);
 
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
     // console.log(trackId, volume, muteBtn)
 
     if (trackId < 0){
       multitrack.wavesurfers.forEach((wavesurfer) => {
         wavesurfer.setVolume(volume);
         if (volume <= 0) {
           container = wavesurfer["renderer"]["container"]
           container.classList.add("disabletrack");
         } else {
           container = wavesurfer["renderer"]["container"]
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
   };
   const syncVideoToTrack = () => {
     currentTime = videoPlayer.currentTime;
     multitrack.setTime(currentTime);
   };
   trackContainer = document.querySelector(`#multitrack${index}`);
   // console.log(trackContainer)
   trackContainer.onclick = () => {
     syncTrackToVideo()
   }
 
 }
 
 
 
 const initMultiTrack = (filenames) => {
   const track_objs = []
 
   for (var f=0; f < filenames.length; f++){
     trackData = []
     for (var i = 0; i < metaData.length; i++) {
       // console.log(`./static/audio/${filename}/${metaData[i]["stem"].toLowerCase()}.wav`)
       track = {
         id: i,
         draggable: false,
         // startPosition: 0.01,
         // startCue: 0.01,
         volume: 0.5,
         options: {
           height: videoHeight/4,
           waveColor: "#656666", // 'hsl(206, 100%, 65%)',
           progressColor: metaData[i]["progressColor"],
         },
         url: `static/audio/${filenames[f][1]}/${metaData[i]["stem"].toLowerCase()}.mp3`,
         intro: {
           endTime: 120,
           label: metaData[i]["stem"],
           color: '#FFE56E',
         }
       }
       trackData.push(track)
     }
     // console.log(`#mutlitrack${f}`)
     multitrack = Multitrack.create(trackData,
       {
         container: document.querySelector(`#multitrack${f}`), // required!
         minPxPerSec: 12, // zoom level
         rightButtonDrag: false, // drag tracks with the right mouse button
         cursorColor: '#57BAB6',
         cursorWidth: 5,
         trackBackground: '#2D2D2D',
         trackBorderColor: '#7C7C7C',
         envelopeOptions: {
           lineColor: 'rgba(255, 0, 0, 0.7)',
           lineWidth: 0,
           dragPointSize: 0,
           dragPointFill: 'rgba(255, 255, 255, 0.8)',
           dragPointStroke: 'rgba(255, 255, 255, 0.3)',
         },
       },
     )
     track_objs.push(multitrack)
   }
 
   return track_objs
 }
 
 
 /* init */
 
 // filenames = [(0, "G5ERdrjBe40"), (1, "fdcJEKh6jFw"), (2, "rRFDUclCpvU")]
 multitrack_objs = initMultiTrack(trackNames)
 // resultContainers = document.querySelectorAll(".result-container")
 
 $(document).ready(function() {
   addController()
   for (let trackId = 0; trackId < multitrack_objs.length; trackId++){
     addEventMultitrack(multitrack_objs[trackId], trackId)
   }
 });