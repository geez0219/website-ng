import { Component, OnInit} from '@angular/core';
import { Title } from '@angular/platform-browser';

declare function startAnimation(): void;
declare function stopAnimation(): void;

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
    startAnimation();
  }

  ngOnDestroy(){
    stopAnimation();
  }
}
