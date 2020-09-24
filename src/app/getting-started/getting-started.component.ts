import { Component, OnInit} from '@angular/core';
import { Title } from '@angular/platform-browser';

declare var tsParticles;

@Component({
  selector: 'app-getting-started',
  templateUrl: './getting-started.component.html',
  styleUrls: ['./getting-started.component.css']
})
export class GettingStartedComponent implements OnInit {
  particleStyle: object = {};
  particleParams: object = {};
  scriptTag:any;
  width: number = 100;
  height: number = 100;

  data = {
    name: "FastEstimator"
  }

  constructor(private title: Title) {}

  ngOnInit() {
    this.title.setTitle(this.data.name);
    tsParticles
    .loadJSON("tsparticles", "./assets/tsparticles.json")
    .then((container) => {
    })
    .catch((error) => {
        console.error(error);
    });
  }


  ngOnDestroy(){
    tsParticles
    .loadJSON("tsparticles", "./assets/tsparticles.json")
    .then((container) => {
        const particles = tsParticles.domItem(0);
        particles.pause();
    })
    .catch((error) => {
        console.error(error);
    });
  }
}
