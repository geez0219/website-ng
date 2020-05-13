import { Component, OnInit} from '@angular/core';
import { Title } from '@angular/platform-browser';
import { Renderer2 } from '@angular/core';

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

  constructor(private title: Title,
              private renderer2: Renderer2) {}

  ngOnInit() {
    this.title.setTitle(this.data.name);
    const s2 = this.renderer2.createElement('script');
    s2.type = 'text/javascript';
    s2.src = './assets/js/tsparticles.js';
    this.renderer2.appendChild(document.body, s2);
  }
}
