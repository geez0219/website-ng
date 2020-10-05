import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Title } from '@angular/platform-browser';

import { GlobalService } from '../global.service';

declare var tsParticles;

@Component({
  selector: 'app-getting-started',
  templateUrl: './getting-started.component.html',
  styleUrls: ['./getting-started.component.css'],
})
export class GettingStartedComponent implements OnInit, OnDestroy {
  particleStyle: object = {};
  particleParams: object = {};
  scriptTag;
  width = 100;
  height = 100;
  currentVersion: string;
  latestVersion: string;

  data = {
    name: 'FastEstimator',
  };

  constructor(private router: Router,
              private title: Title,
              private globalService: GlobalService) {}

  ngOnInit() {
    this.latestVersion = this.globalService.getLatestVersion();
    this.currentVersion = this.globalService.getSelectedVersion();
    this.globalService.version.subscribe((v: string) => {
      this.currentVersion = v;
    });
    this.title.setTitle(this.data.name);
    tsParticles
      .loadJSON('tsparticles', './assets/tsparticles.json')
      .then((container) => {})
      .catch((error) => {
        console.error(error);
      });
  }

  ngOnDestroy() {
    tsParticles
      .loadJSON('tsparticles', './assets/tsparticles.json')
      .then((container) => {
        const particles = tsParticles.domItem(0);
        particles.pause();
      })
      .catch((error) => {
        console.error(error);
      });
  }

  navigateToLatestExample(exampleURL: string) {
    this.globalService.version.next(this.latestVersion);
    this.router.navigate([exampleURL]);
  }
}
