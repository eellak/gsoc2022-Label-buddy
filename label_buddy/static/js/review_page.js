var selected_label = null;
var selected_label_color = null;
var selected_region = null;
var selected_region_button = null;
var regions_count = 0;
var color_when_selected = '#74deed';
var initial_opacity = .2;
var selected_region_opacity = .9;
var wavesurfer; // eslint-disable-line no-var

function toggleIcon(button){
    $(button).find('i').remove();
    if (wavesurfer.backend.isPaused()) {
        $(button).html($('<i/>',{class:'fas fa-play'})).append(' Play');
    }
    else {
        $(button).html($('<i/>',{class:'fas fa-pause'})).append(' Pause');
    }
}

// same as annotation page but removed drug feature
function selectedLabel(button) {

    if(selected_label == button) {
        
        // change cursor
        selected_label.style.cursor = 'pointer';

        selected_label.style.opacity = initial_opacity;
        selected_label = null;
        selected_label_color = null;
    } else {
        // if label already selected, unselect it
        if(selected_label) {
            // change cursor
            selected_label.style.cursor = 'pointer';

            selected_label.style.opacity = initial_opacity;
        }

        // set new selected label
        selected_label = button;
        selected_label_color = button.style.backgroundColor;
        selected_label.style.opacity = 1;

        // change cursor
        selected_label.style.cursor = 'auto';
    }
}

function selectRegionButton(button) {
    let region_by_id = wavesurfer.regions.list[button.id];
    if(selected_region_button == button) {
        // change cursor
        selected_region_button.style.cursor = 'pointer';

        selected_region_button.style.opacity = initial_opacity;
        selected_region_button.style.fontWeight = 'normal';
        selected_region_button.style.backgroundColor = 'rgb(245,245,245)'
        selected_region_button = null;

        // if region selected, unselect it
        if(selected_region == region_by_id) {
            region_by_id.wavesurfer.fireEvent('region-click', region_by_id);
        }
    } else {
        // if label already selected, unselect it
        if(selected_region_button) {
            
            // change cursor
            selected_region_button.style.cursor = 'pointer';

            selected_region_button.style.opacity = initial_opacity;
            selected_region_button.style.fontWeight = 'normal';
            selected_region_button.style.backgroundColor = 'rgb(245,245,245)';
            // if region selected, unselect it
            if(selected_region == region_by_id) {
                region_by_id.wavesurfer.fireEvent('region-click', region_by_id);
            }
        }
        
        selected_region_button = button;
        selected_region_button.style.opacity = 1;
        selected_region_button.style.backgroundColor = '#dddddd';
        selected_region_button.style.fontWeight = 'bold';

        if(selected_region != region_by_id) {
            region_by_id.wavesurfer.fireEvent('region-click', region_by_id);
        }

        // change cursor
        selected_region_button.style.cursor = 'auto';
    }

}

function hoverRegionButtonIn(button) {
    button.style.opacity = selected_region_opacity;
    button.style.backgroundColor = '#dddddd';
    let region_by_id = wavesurfer.regions.list[button.id];
    region_by_id.update({
        color: rgbToRgba(region_by_id.data['color'], selected_region_opacity)
    });
    getLabelButton(region_by_id.data['label']).style.opacity = selected_region_opacity;
}

function hoverRegionButtonOut(button) {
    let region_by_id = wavesurfer.regions.list[button.id];
    if(selected_region_button != button) {
        button.style.opacity = initial_opacity;
        button.style.backgroundColor = 'rgb(245,245,245)';
        region_by_id.update({
            color: rgbToRgba(region_by_id.data['color'], initial_opacity)
        });
    }

    let lbl_button = getLabelButton(region_by_id.data['label']);
    if(selected_label != lbl_button) {
        lbl_button.style.opacity = initial_opacity;
    }
}

// rgb to rgba with opacity 0.1
function rgbToRgba(rgb, opacity) {
    if(rgb.indexOf('a') == -1){
        var rgba = rgb.replace(')', ', ' + opacity + ')').replace('rgb', 'rgba');
    }
    return rgba;
}

// get label to select after region click
function getLabelButton(label) {
    var alllabels = document.getElementsByClassName('my-badge');
    for(const x of alllabels) {
        if(x.value == label) {
            return x;
        }
    }
}

function getLabelColorByValue(label)
{
    var alllabels = document.getElementsByClassName('my-badge');
    for(const x of alllabels) {
        if(x.value == label) {
            return x.style.backgroundColor;
        }
    }
}

function getRegionButton(new_region) {
    let new_region_button = document.createElement('BUTTON');
    // set attributes
    new_region_button.className = 'region-buttons';
    // new_region_button.style.backgroundColor = new_region.data['color'];
    new_region_button.style.opacity = initial_opacity;
    new_region_button.id = new_region.id;
    new_region_button.title = "Label: " + new_region.data['label'];

   
    new_region_button.setAttribute( "onClick", "selectRegionButton(this);" );
    new_region_button.setAttribute( "onmouseover", "hoverRegionButtonIn(this);" );
    new_region_button.setAttribute( "onmouseout", "hoverRegionButtonOut(this);" );

    let count = document.createElement("SPAN");
    count.id = "count";
    count.style.display = 'inline-block';
    count.style.width = '35px'
    count.style.marginRight = '7px'
    
    let icon = document.createElement('i');
    icon.className = 'fas fa-music';
    icon.style.marginRight = "10px";
    icon.style.color = new_region.data['color'];

    let timings = document.createElement("SPAN");
    timings.id = "timings";

    count.textContent = regions_count + ".";
    timings.textContent = (Math.round((new_region.start + Number.EPSILON) * 100) / 100) + " - " + (Math.round((new_region.end + Number.EPSILON) * 100) / 100);
    new_region_button.appendChild(count);
    new_region_button.appendChild(icon);
    new_region_button.appendChild(timings);
    return new_region_button;
}

function add_region_to_section(region) {
    // load region to region section
    let new_region_button = getRegionButton(region);
    $('#regions-div').append(new_region_button);
}

//----------------------------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function() {
    // Init wavesurfer
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        backend: 'MediaElement',
        height: 250,
        pixelRatio: 1,
        scrollParent: true,
        normalize: true,
        splitChannels: false,
        waveColor: '#ddd',
        progressColor: '#ddd',
        plugins: [
            WaveSurfer.regions.create({}),
            WaveSurfer.timeline.create({
                container: '#wave-timeline'
            }),
            WaveSurfer.cursor.create({
                showTime: true,
                opacity: 1,
                customShowTimeStyle: {
                    'background-color': '#000',
                    color: '#fff',
                    padding: '2px',
                    'font-size': '15px'
                }
            }),
        ]
    });

    wavesurfer.load(audio_url, JSON.parse(audio_waveform_data), 'auto');

    /* Regions */

    // load regions of existing annotation (if exists)
    wavesurfer.on('ready', function() {
        wavesurfer.setPlaybackRate(1);
        wavesurfer.zoom(0); // initial zoom
        wavesurfer.setVolume(1); // initial volume
        let result = annotation;
        // if there is a result load regions of annotation
        if(result && result.length != 0) {
            loadRegions(result);
        }
    });

    // audioprocess as the audio is playing - calculate the loaded percentage each time
    wavesurfer.on('audioprocess', function() {
        let loaded_percent = get_loaded_precentage(wavesurfer.backend.media);
        if (loaded_percent < 1){
            NProgress.set(loaded_percent); 
        }
        if (loaded_percent == 1){
            NProgress.done();
        }
    });


    // on region click select it
    wavesurfer.on('region-click', function(region) {
        let region_button = document.getElementById(region.id);

        // if region already selected, unselect it
        if(selected_region == region) {
            region.update({
                color: rgbToRgba(region.data['color'], initial_opacity)
            });
            selected_region = null;
            document.getElementById('play-region-btn').style.display = 'none';

            // deactivate label of region
            let lbl = getLabelButton(region.data['label']);
            if(selected_label == lbl) lbl.click();

            // deactivate region button
            if(selected_region_button == region_button) {
                region_button.click();
            }

        } else {

            // if another region is selected, unselect it
            if(selected_region) {
                selected_region.update({
                    color: rgbToRgba(selected_region.data['color'], initial_opacity)
                });

                // deactivate label of region
                let lbl = getLabelButton(selected_region.data['label']);
                if(selected_label == lbl) lbl.click();

                // deactivate region button
                if(selected_region_button == region_button) {
                    region_button.click();
                }
            }

            region.update({
                color: rgbToRgba(region.data['color'], selected_region_opacity)
            });
            selected_region = region;
            document.getElementById('play-region-btn').style.display = 'inline';

            // activate label of region
            let lbl = getLabelButton(region.data['label']);
            if(selected_label != lbl) lbl.click();

            // activate region button
            if(selected_region_button != region_button) {
                region_button.click();
            }
        }
    });


    // on region created set its data to current label
    wavesurfer.on('region-created', function(region) {
        // increase counter
        regions_count++;
        add_region_to_section(region);
    });

    // when audio finishes, toggle play/pause button and hide scrollbar
    wavesurfer.on('finish', function() {
        wavesurfer.stop()
    });

    // on play or pause toggle play/pause button
    wavesurfer.on('play', function() {
        toggleIcon(document.getElementById('play-pause-button'));
    });
    wavesurfer.on('pause', function() {
        toggleIcon(document.getElementById('play-pause-button'));
    });

});

function showAlert() {
    NProgress.done();
    location.reload(true);
}

// Load regions from annotation.
function loadRegions(result) {
    for(const region of result) {
        wavesurfer.addRegion({
            start: region['value']['start'],
            end: region['value']['end'],
            loop: false,
            color: rgbToRgba(getLabelColorByValue(region['value']['label']), initial_opacity),
            resize: false,
            drag: false,
            data: {
                label: region['value']['label'],
                color: getLabelColorByValue(region['value']['label'])
            }
        });
    }
}

function submitReview(button) {
    // xmlhttp request for exporting data
    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            // Typical action to be performed when the document is ready:
            showAlert();
        } else if(this.readyState == 4 && (this.status == 400 || this.status == 401)) {
            showAlert();
        }
    };
    xhr.open("POST", "", true);
    xhr.setRequestHeader("X-CSRFToken", django_csrf_token);
    xhr.setRequestHeader("Content-Type", "application/json");
    let data = {
        "value": button.value,
        "comment": $('#commentArea').val(),
    };
    NProgress.start();
    xhr.send(JSON.stringify(data));
}

// buttons

// play region link
$('#play-region-btn').click( function(e) {
    e.preventDefault(); 
    if(selected_region){
        selected_region.play();
    }
    return false; 
} );

document.getElementById('zoom-slider').oninput = function () {
    wavesurfer.zoom(Number(this.value));
};

function toggleSoundIcon(slider){
    wavesurfer.setVolume(Number(slider.value));
    let sound_slider_icon = document.getElementById('sound-slider-icon');
    if(slider.value > 0 && slider.value <= .5) {
        sound_slider_icon.classList.toggle('fa-volume-down');
    } else if(slider.value > .5){
        sound_slider_icon.classList.toggle('fa-volume-up');
    } else {
        sound_slider_icon.classList.toggle('fa-volume-mute');
    }
};

// mute unmute button
$('#mute-unmute-btn').click( function(e) {
    e.preventDefault(); 
    let current_volume = wavesurfer.getVolume();
    let sound_slider = document.getElementById('sound-slider');
    if(current_volume > 0) {
        // then mute
        sound_slider.value = 0;
        toggleSoundIcon(sound_slider);
    } else {
        sound_slider.value = .5;
        toggleSoundIcon(sound_slider);
    }
    return false; 
} );


function changeSpeed(selector) {
    wavesurfer.setPlaybackRate(selector.value);
}
