function stopAnimation() {
    tsParticles
        .loadJSON("tsparticles", "./assets/tsparticles.json")
        .then((container) => {
            console.log("callback - tsparticles config loaded");
            const instance = tsParticles.domItem(0);
            instance.pause();
        })
        .catch((error) => {
            console.error(error);
        });
}

function startAnimation() {
    tsParticles
        .loadJSON("tsparticles", "./assets/tsparticles.json")
        .then((container) => {
            console.log("callback - tsparticles config loaded");
        })
        .catch((error) => {
            console.error(error);
        });
}