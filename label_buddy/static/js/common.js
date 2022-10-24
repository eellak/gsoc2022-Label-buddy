$(document).ready(function(){
    $('.clickable-row').click(function(){
        window.location = $(this).attr('href');
        return false;
    });
});

$(window).load(function(){
    NProgress.done();
 });
 
 $(document).ready(function() {
    NProgress.start();
 });

document.addEventListener('DOMContentLoaded', function() {
    $('.tooltip-icons').data('title'); // "This is a test";

    $(function () {
        $('[data-toggle="tooltip"]').tooltip({
            trigger : 'hover'
        })
    });

    $('form').on('submit',function(){
        NProgress.start();
    });
});


function get_loaded_precentage(audio) {
    var buffered = audio.buffered; // returns the buffered portion of the audio
    var loaded; // the loaded portion of the audio

    loaded = buffered.end(0) / audio.duration;  // calculate the loaded percent of the audio

    return loaded;
}
