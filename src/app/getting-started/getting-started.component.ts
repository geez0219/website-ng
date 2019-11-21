import { Component, OnInit } from '@angular/core';
import { Title } from '@angular/platform-browser';

@Component({
  selector: 'app-getting-started',
  templateUrl: './getting-started.component.html',
  styleUrls: ['./getting-started.component.css']
})
export class GettingStartedComponent implements OnInit {

  data = {
    name: "FastEstimator"
  }

  constructor(private title: Title) { }

  ngOnInit() {
    this.title.setTitle(this.data.name);
  }

}
