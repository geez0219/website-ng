import { Component, OnInit, ViewEncapsulation } from '@angular/core';
import { Location } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute } from '@angular/router';

import { Tutorial } from '../tutorial';

@Component({
  selector: 'app-examples',
  templateUrl: './examples.component.html',
  styleUrls: ['../tutorial/tutorial.component.css']
})
export class ExamplesComponent implements OnInit {
  selectedTutorial: string;
  currentSelection: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];

  constructor(private http: HttpClient,
    private route: ActivatedRoute,
    private location: Location) { }

  ngOnInit() {
    this.getTutorialStructure();
  }

  getTutorialStructure() {
    this.http.get('assets/apphub/structure.json', {responseType: 'text'}).subscribe(data => {
      this.tutorialList = <Tutorial[]>JSON.parse(data);

      let name = this.route.snapshot.paramMap.get('name');
      if (name === null) {
        this.updateCurrentTutorial(this.tutorialList[0]);
      } else {
        var t: Tutorial[] = this.tutorialList.filter(tutorial => tutorial.name === (name + ".md"));
        this.updateCurrentTutorial(t[0]);
      }
      
    });
    // console.log(typeof(this.tutorialList));
  }

  updateCurrentTutorial(tutorial: Tutorial) {
    window.scroll(0,0);
    this.selectedTutorial = tutorial.name;
    this.currentSelection = 'assets/apphub/' + this.selectedTutorial;

    this.getSelectedTutorialText();

    this.location.replaceState('/examples/' + tutorial.name.substring(0, tutorial.name.length - 3));
  }

  getSelectedTutorialText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

}

